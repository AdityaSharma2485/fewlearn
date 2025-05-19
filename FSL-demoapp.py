import streamlit as st
import time
import random
import os
import torch
from torch import nn, optim
import torchvision.models as models
from torch.utils.data import DataLoader
from easyfsl.samplers import TaskSampler
import tempfile
import zipfile
from tkinter import Tk, filedialog
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import from restructured fewlearn module
from fewlearn.core.minds import MINDS
from fewlearn.models.prototypical import PrototypicalNetworks
from fewlearn.evaluation.evaluator import Evaluator
from fewlearn.visualization.plotting import plot_task_examples

def update_progress(progress_bar, step):
    """Update the progress bar."""
    progress_bar.progress(step)

def temp_load(test_set, N_WAY, N_SHOT, N_QUERY, N_EVALUATION_TASKS, progress_bar, data = 'custom'):
    """
    Load the test set and create a DataLoader using TaskSampler.

    Args:
        test_set: The dataset to sample from.
        N_WAY: Number of classes for each task.
        N_SHOT: Number of examples per class for the support set.
        N_QUERY: Number of examples per class for the query set.
        N_EVALUATION_TASKS: Number of evaluation tasks.
        progress_bar: Streamlit progress bar for visual feedback.
        data: Dataset type ('custom' or 'omniglot')

    Returns:
        DataLoader: DataLoader object for the test set.
    """
    if data == 'omniglot':
        test_set.get_labels = lambda: [instance[1] for instance in test_set._flat_character_images]
    else:
        test_set.get_labels = lambda: test_set.labels
    # Progress update
    update_progress(progress_bar, 20)
    
    test_sampler = TaskSampler(
        test_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
    )

    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )
    
    # Progress update after test loader is ready
    update_progress(progress_bar, 60)
    
    return test_loader

# Main page heading with larger font size for title and subtitle
st.markdown("<h1 style='text-align: center; font-size: 5em; margin-bottom: 0.1em;'>M.I.N.D.S</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; font-size: 2.5em; margin-top: -20px;'>Minimal Instance Neural Data System</h2>", unsafe_allow_html=True)

# Placeholder for loading spinner
placeholder = st.empty()

with placeholder.container():
    with st.spinner("Loading libraries and initializing..."):
        # Create a dictionary of available backbone models
        available_backbones = {
            "GoogleNet": models.googlenet(weights="DEFAULT"),
            "ResNet18": models.resnet18(weights="DEFAULT"),
            "ResNet50": models.resnet50(weights="DEFAULT"),
            "MobileNetV2": models.mobilenet_v2(weights="DEFAULT"),
            "DenseNet121": models.densenet121(weights="DEFAULT"),
        }
        
        # Modify the fully connected layer to output features
        for name, model in available_backbones.items():
            if hasattr(model, 'fc'):
                model.fc = nn.Flatten()
            elif hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Sequential):
                    model.classifier = nn.Flatten()
                else:
                    model.classifier = nn.Flatten()
                    
        st.success("Libraries and initialization complete!")

placeholder.empty()

def main_page():
    pass

def omniglot_dataset():
    # Constants
    N_QUERY = 10  # Number of images per class in the query set
    N_EVALUATION_TASKS = 100
    
    # Initialize the selector for backbone models
    st.sidebar.subheader("Select Models to Evaluate")
    selected_models = {}
    for model_name in available_backbones.keys():
        if st.sidebar.checkbox(model_name, value=(model_name == "GoogleNet")):
            selected_models[model_name] = available_backbones[model_name]
    
    # Initialize with selected models or at least one model if none selected
    if not selected_models:
        selected_models = {"GoogleNet": available_backbones["GoogleNet"]}
        st.sidebar.warning("At least one model must be selected. Using GoogleNet by default.")
    
    # Initialize MINDS class and add the selected models
    meta_eval = MINDS()
    for name, backbone in selected_models.items():
        proto_model = PrototypicalNetworks(backbone=backbone)
        meta_eval.add_model(name, proto_model)

    def load_dataset():
        try:
            # Initialize the progress bar
            progress_bar = st.progress(0)

            # Load dataset
            _, test_set = meta_eval.load_omniglot(image_size=128)
            progress_bar.progress(40)

            # Prepare test loader
            st.session_state.test_loader = temp_load(test_set=test_set, N_SHOT=N_SHOT, N_WAY=N_WAY, N_QUERY=N_QUERY, N_EVALUATION_TASKS=N_EVALUATION_TASKS, data='omniglot', progress_bar=progress_bar)
            progress_bar.progress(100)  # Complete progress
            st.session_state.test_set = test_set  # Store test_set in session state
            st.success("Dataset Loaded!")

        except FileNotFoundError:
            st.error("The dataset could not be found. Please check the file path.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    def visualize():
        # Initialize the progress bar
        progress_bar = st.progress(0)

        # Fetch example support and query images
        (
            example_support_images,
            example_support_labels,
            example_query_images,
            example_query_labels,
            example_class_ids,
        ) = next(iter(st.session_state.test_loader))

        progress_bar.progress(80)

        # Visualization of support and query set images using the plotting module
        st.session_state.support_plot = plot_task_examples(
            example_support_images, 
            example_support_labels, 
            max_examples=N_SHOT
        )
        
        st.session_state.query_plot = plot_task_examples(
            example_query_images, 
            example_query_labels, 
            max_examples=N_QUERY
        )

        # Complete progress
        progress_bar.progress(100)
        st.success("Visualization Complete!")

        # Show Support Set & Query Set:
        support_set, query_set = st.columns(2)

        with support_set:
            st.write("### Support Set")
            st.pyplot(st.session_state.support_plot)

        with query_set:
            st.write("### Query Set")
            st.pyplot(st.session_state.query_plot)

    def evaluate_support_query_sets():
        if 'test_set' not in st.session_state or st.session_state.test_loader is None:
            st.error("Please load the OMNIGLOT dataset before evaluating.")
            return

        st.success('Evaluation has begun...')
        
        # Get the list of selected models
        selected_model_names = list(meta_eval.models.keys())
        num_models = len(selected_model_names)
        
        # Display which models are being evaluated
        st.write(f"Evaluating {num_models} models in parallel: {', '.join(selected_model_names)}")
        
        # Initialize the progress bar
        progress_bar = st.progress(0)
        
        # Create a progress status message
        status_text = st.empty()
        status_text.text(f"Starting evaluation of {num_models} models...")
        
        try:
            # Call evaluate function and store metrics
            all_metrics, all_inference = meta_eval.evaluate(
                st.session_state.test_loader,
                n_tasks=N_EVALUATION_TASKS,
                bar=False,  # Disable console bar since we're using Streamlit's progress tracking
                webview_bar=progress_bar
            )
            
            progress_bar.progress(100)  # Complete progress
            status_text.text("Evaluation Complete!")
            
            # Display results in a table format
            results_df = pd.DataFrame({
                'Model': [],
                'Accuracy': [],
                'F1 Score': [],
                'Inference Time (s)': []
            })
            
            for model_name, metrics in all_metrics.items():
                new_row = pd.DataFrame({
                    'Model': [model_name],
                    'Accuracy': [f"{metrics['accuracy']:.2%}"],
                    'F1 Score': [f"{metrics['f1']:.4f}"],
                    'Inference Time (s)': [f"{all_inference[model_name]:.4f}"]
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)
            
            st.table(results_df)
            
            # Visualization of confusion matrices
            if 'confusion_matrix' in all_metrics[selected_model_names[0]]:
                st.write("### Confusion Matrices")
                
                cols = st.columns(min(3, num_models))
                for i, model_name in enumerate(selected_model_names):
                    with cols[i % min(3, num_models)]:
                        st.write(f"**{model_name}**")
                        cm = all_metrics[model_name]['confusion_matrix']
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
                        plt.colorbar(im, ax=ax)
                        plt.title(f"{model_name} Confusion Matrix")
                        
                        st.pyplot(fig)
        
        except Exception as e:
            st.error(f"An error occurred during evaluation: {e}")

    # Set up the form for configuring the experiment parameters
    with st.form(key='omniglot_form'):
        st.write("## Omniglot Dataset Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            N_WAY = st.slider("Number of classes (N-way)", min_value=2, max_value=20, value=5)
        
        with col2:
            N_SHOT = st.slider("Number of support examples per class (N-shot)", min_value=1, max_value=10, value=5)
        
        # Create the submit button
        submit_button = st.form_submit_button(label='Load Dataset')
    
    # Execute the loading if the submit button was pressed
    if submit_button:
        load_dataset()
    
    # Only show the buttons if the dataset is loaded
    if 'test_loader' in st.session_state:
        button_cols = st.columns(2)
        
        with button_cols[0]:
            if st.button("Visualize Dataset"):
                visualize()
        
        with button_cols[1]:
            if st.button("Evaluate Models"):
                evaluate_support_query_sets()

def custom_dataset():
    # Setup params
    N_SHOT = 5  # Number of examples per class in the support set
    N_QUERY = 10  # Number of images per class in the query set
    N_WAY = 5  # Number of classes in the task
    N_EVALUATION_TASKS = 100
    CUSTOM_DATASET_STRUCTURE = """
    custom-dataset/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2/
    │   ├── image1.jpg
    │   └── ...
    └── ...
    """
    st.write("## Custom Dataset")
    
    def display_folder_structure():
        # Folder structure example using markdown
        st.markdown("### Expected Folder Structure")
        st.text(CUSTOM_DATASET_STRUCTURE)
    
    display_folder_structure()
    
    # Initialize the selector for backbone models
    st.sidebar.subheader("Select Models to Evaluate")
    selected_models = {}
    for model_name in available_backbones.keys():
        if st.sidebar.checkbox(model_name, value=(model_name == "GoogleNet")):
            selected_models[model_name] = available_backbones[model_name]
    
    # Initialize with selected models or at least one model if none selected
    if not selected_models:
        selected_models = {"GoogleNet": available_backbones["GoogleNet"]}
        st.sidebar.warning("At least one model must be selected. Using GoogleNet by default.")
    
    # Initialize MINDS class and add the selected models
    meta_eval = MINDS()
    for name, backbone in selected_models.items():
        proto_model = PrototypicalNetworks(backbone=backbone)
        meta_eval.add_model(name, proto_model)
    
    def load_customdata(uploaded_file):
        try:
            # Initialize the progress bar
            progress_bar = st.progress(0)
            
            # Create a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save the uploaded file to the temporary directory
                zip_path = os.path.join(temp_dir, "dataset.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Extract the ZIP file
                extract_dir = os.path.join(temp_dir, "extracted")
                os.makedirs(extract_dir, exist_ok=True)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Load dataset using MINDS custom dataset loader
                test_set = meta_eval.load_custom_dataset(extract_dir)
                progress_bar.progress(40)
                
                # Prepare test loader
                st.session_state.test_loader = temp_load(
                    test_set=test_set, 
                    N_SHOT=N_SHOT, 
                    N_WAY=N_WAY, 
                    N_QUERY=N_QUERY, 
                    N_EVALUATION_TASKS=N_EVALUATION_TASKS,
                    progress_bar=progress_bar
                )
                
                progress_bar.progress(100)  # Complete progress
                st.session_state.test_set = test_set  # Store test_set in session state
                st.success("Custom Dataset Loaded!")
                
                # Display some information about the dataset
                st.write(f"Dataset contains {len(test_set.classes)} classes:")
                st.write(", ".join(test_set.classes[:10]) + ("..." if len(test_set.classes) > 10 else ""))
                st.write(f"Total number of images: {len(test_set)}")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
    
    def visualize():
        # Initialize the progress bar
        progress_bar = st.progress(0)

        # Fetch example support and query images
        (
            example_support_images,
            example_support_labels,
            example_query_images,
            example_query_labels,
            example_class_ids,
        ) = next(iter(st.session_state.test_loader))

        progress_bar.progress(80)

        # Visualization of support and query set images using the plotting module
        st.session_state.support_plot = plot_task_examples(
            example_support_images, 
            example_support_labels, 
            max_examples=N_SHOT
        )
        
        st.session_state.query_plot = plot_task_examples(
            example_query_images, 
            example_query_labels, 
            max_examples=N_QUERY
        )

        # Complete progress
        progress_bar.progress(100)
        st.success("Visualization Complete!")

        # Show Support Set & Query Set:
        support_set, query_set = st.columns(2)

        with support_set:
            st.write("### Support Set")
            st.pyplot(st.session_state.support_plot)

        with query_set:
            st.write("### Query Set")
            st.pyplot(st.session_state.query_plot)
    
    def evaluate_support_query_sets():
        if 'test_set' not in st.session_state or st.session_state.test_loader is None:
            st.error("Please load a custom dataset before evaluating.")
            return

        st.success('Evaluation has begun...')
        
        # Get the list of selected models
        selected_model_names = list(meta_eval.models.keys())
        num_models = len(selected_model_names)
        
        # Display which models are being evaluated
        st.write(f"Evaluating {num_models} models in parallel: {', '.join(selected_model_names)}")
        
        # Initialize the progress bar
        progress_bar = st.progress(0)
        
        # Create a progress status message
        status_text = st.empty()
        status_text.text(f"Starting evaluation of {num_models} models...")
        
        try:
            # Call evaluate function and store metrics
            all_metrics, all_inference = meta_eval.evaluate(
                st.session_state.test_loader,
                n_tasks=N_EVALUATION_TASKS,
                bar=False,  # Disable console bar since we're using Streamlit's progress tracking
                webview_bar=progress_bar
            )
            
            progress_bar.progress(100)  # Complete progress
            status_text.text("Evaluation Complete!")
            
            # Display results in a table format
            results_df = pd.DataFrame({
                'Model': [],
                'Accuracy': [],
                'F1 Score': [],
                'Inference Time (s)': []
            })
            
            for model_name, metrics in all_metrics.items():
                new_row = pd.DataFrame({
                    'Model': [model_name],
                    'Accuracy': [f"{metrics['accuracy']:.2%}"],
                    'F1 Score': [f"{metrics['f1']:.4f}"],
                    'Inference Time (s)': [f"{all_inference[model_name]:.4f}"]
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)
            
            st.table(results_df)
            
            # Visualization of confusion matrices
            if 'confusion_matrix' in all_metrics[selected_model_names[0]]:
                st.write("### Confusion Matrices")
                
                cols = st.columns(min(3, num_models))
                for i, model_name in enumerate(selected_model_names):
                    with cols[i % min(3, num_models)]:
                        st.write(f"**{model_name}**")
                        cm = all_metrics[model_name]['confusion_matrix']
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
                        plt.colorbar(im, ax=ax)
                        plt.title(f"{model_name} Confusion Matrix")
                        
                        st.pyplot(fig)
        
        except Exception as e:
            st.error(f"An error occurred during evaluation: {e}")
    
    # Set up the form for configuring the experiment parameters
    with st.form(key='custom_dataset_form'):
        st.write("## Custom Dataset Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            N_WAY = st.slider("Number of classes (N-way)", min_value=2, max_value=20, value=5)
        
        with col2:
            N_SHOT = st.slider("Number of support examples per class (N-shot)", min_value=1, max_value=10, value=5)
        
        uploaded_file = st.file_uploader("Upload Custom Dataset (ZIP file)", type="zip")
        
        # Create the submit button
        submit_button = st.form_submit_button(label='Load Dataset')
    
    # Execute the loading if the submit button was pressed and a file was uploaded
    if submit_button and uploaded_file is not None:
        load_customdata(uploaded_file)
    
    # Only show the buttons if the dataset is loaded
    if 'test_loader' in st.session_state:
        button_cols = st.columns(2)
        
        with button_cols[0]:
            if st.button("Visualize Dataset"):
                visualize()
        
        with button_cols[1]:
            if st.button("Evaluate Models"):
                evaluate_support_query_sets()

# Sidebar navigation
pages = {
    "Main Page": main_page,
    "Omniglot Dataset": omniglot_dataset,
    "Custom Dataset": custom_dataset
}

# Add styling to the sidebar
st.sidebar.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", list(pages.keys()))
pages[selected_page]()
