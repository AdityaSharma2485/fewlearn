import streamlit as st
import os
import torch
import numpy as np
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader
from easyfsl.samplers import TaskSampler
import tempfile
import zipfile
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Import from restructured fewlearn module
from fewlearn.core.minds import MINDS
from fewlearn.models.prototypical import PrototypicalNetworks
from fewlearn.evaluation.evaluator import Evaluator
from fewlearn.visualization import plot_support_query_sets

# Main page heading with styling improvements
st.markdown("<h1 style='text-align: center; color: #2c3e50; font-size: 3.5em; margin-bottom: 0.1em;'>M.I.N.D.S</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #34495e; font-size: 1.8em; margin-top: -10px;'>Minimal Instance Neural Data System</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d; font-style: italic;'>Few-Shot Learning Evaluation Framework</p>", unsafe_allow_html=True)

# CSS for layout, button styling, and spacing adjustments
st.markdown("""
    <style>
    /* General background and text color for the app */
    .stApp {
        background-color: #f8f9fa;
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Improve sidebar appearance */
    .css-1d391kg {
        background-color: #e8eef2;
    }
    
    /* Style buttons */
    .stButton button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background-color: #2980b9;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Style headers */
    h3 {
        color: #2c3e50;
        border-bottom: 1px solid #eee;
        padding-bottom: 10px;
    }
    
    /* Style metrics display */
    .metric-container {
        background-color: #f1f8fe;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        border-left: 3px solid #3498db;
    }
    </style>
""", unsafe_allow_html=True)

# Create backbone dictionary
@st.cache_resource
def load_backbones():
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
    
    return available_backbones

# Helper function to update progress bars
def update_progress(progress_bar, step):
    progress_bar.progress(step)

# Helper function to load test data
def load_test_data(test_set, n_way, n_shot, n_query, n_tasks, progress_bar, data_type='custom'):
    if data_type == 'omniglot':
        test_set.get_labels = lambda: [instance[1] for instance in test_set._flat_character_images]
    else:
        test_set.get_labels = lambda: test_set.labels
        
    update_progress(progress_bar, 20)
    
    test_sampler = TaskSampler(
        test_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks
    )

    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )
    
    update_progress(progress_bar, 60)
    
    return test_loader

# Define the main page function
def main_page():
    st.write("### About Few-Shot Learning")
    st.markdown("""
    Few-Shot Learning allows models to learn from minimal examples, similar to human learning.
    
    This application demonstrates few-shot learning using Prototypical Networks on:
    - **Omniglot dataset**: A collection of handwritten characters from different alphabets
    - **Custom dataset**: Upload your own images organized in class folders
    
    Select a dataset from the sidebar to begin exploring few-shot learning.
    """)
    
    st.image("https://miro.medium.com/max/1400/1*KRywqWzDOvnfkO0IYWdISw.png", 
             caption="Illustration of Few-Shot Learning Concept")

# Define the Omniglot dataset page
def omniglot_dataset():
    st.write("## Omniglot Dataset Evaluation")
    
    # Constants
    N_QUERY = 10
    N_EVALUATION_TASKS = 100
    
    # Sidebar controls
    st.sidebar.subheader("Model Configuration")
    N_WAY = st.sidebar.slider("N-Way (Classes)", min_value=2, max_value=20, value=5)
    N_SHOT = st.sidebar.slider("N-Shot (Examples per class)", min_value=1, max_value=10, value=5)
    
    # Select models to evaluate
    available_backbones = load_backbones()
    st.sidebar.subheader("Select Models to Evaluate")
    selected_models = {}
    for model_name in available_backbones.keys():
        if st.sidebar.checkbox(model_name, value=(model_name == "ResNet18")):
            selected_models[model_name] = available_backbones[model_name]
    
    # Initialize with selected models or at least one model if none selected
    if not selected_models:
        selected_models = {"ResNet18": available_backbones["ResNet18"]}
        st.sidebar.warning("At least one model must be selected. Using ResNet18 by default.")
    
    # Initialize MINDS class and add the selected models
    meta_eval = MINDS()
    for name, backbone in selected_models.items():
        proto_model = PrototypicalNetworks(backbone=backbone)
        meta_eval.add_model(name, proto_model)
    
    # Load dataset button
    if st.button("Load Omniglot Dataset"):
        try:
            progress_bar = st.progress(0)
            _, test_set = meta_eval.load_omniglot(image_size=128)
            progress_bar.progress(40)
            
            st.session_state.test_loader = load_test_data(
                test_set=test_set, 
                n_way=N_WAY, 
                n_shot=N_SHOT, 
                n_query=N_QUERY, 
                n_tasks=N_EVALUATION_TASKS, 
                data_type='omniglot', 
                progress_bar=progress_bar
            )
            
            progress_bar.progress(100)
            st.session_state.test_set = test_set
            st.success("Omniglot Dataset Loaded Successfully!")
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    
    # Visualize support and query sets
    if st.button("Visualize Examples") and 'test_loader' in st.session_state:
        progress_bar = st.progress(0)
        
        # Get a batch from the dataloader
        (
            example_support_images,
            example_support_labels,
            example_query_images,
            example_query_labels,
            example_class_ids,
        ) = next(iter(st.session_state.test_loader))
        
        progress_bar.progress(50)
        
        # Create visualizations ensuring proper handling of image channels
        try:
            support_plot = plot_support_query_sets(
                example_support_images, 
                example_support_labels,
                max_examples=N_SHOT
            )
            query_plot = plot_support_query_sets(
                example_query_images,
                example_query_labels,
                max_examples=N_QUERY
            )
            
            progress_bar.progress(100)
            
            # Show support and query sets
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Support Set")
                st.pyplot(support_plot)
            
            with col2:
                st.write("### Query Set")
                st.pyplot(query_plot)
        except Exception as e:
            import traceback
            st.error(f"Error visualizing examples: {str(e)}")
            st.code(traceback.format_exc(), language="python")
    
    # Evaluate models
    if st.button("Evaluate Models") and 'test_loader' in st.session_state:
        st.write("### Evaluation Results")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        selected_model_names = list(meta_eval.models.keys())
        status_text.text(f"Evaluating {len(selected_model_names)} models: {', '.join(selected_model_names)}")
        
        try:
            # Run evaluation
            all_metrics, all_inference = meta_eval.evaluate(
                st.session_state.test_loader,
                n_tasks=N_EVALUATION_TASKS,
                bar=False,
                webview_bar=progress_bar
            )
            
            progress_bar.progress(100)
            status_text.text("Evaluation Complete!")
            
            # Display results
            for model_name, metrics in all_metrics.items():
                with st.expander(f"{model_name} Results", expanded=True):
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                    with cols[1]:
                        st.metric("F1 Score", f"{metrics['f1']:.4f}")
                    with cols[2]:
                        st.metric("Inference Time", f"{all_inference[model_name]:.4f}s")
                    
                    # Show confusion matrix if available
                    if 'confusion_matrix' in metrics:
                        st.write("#### Confusion Matrix")
                        fig, ax = plt.subplots(figsize=(10, 8))
                        cax = ax.matshow(metrics['confusion_matrix'], cmap='Blues')
                        fig.colorbar(cax)
                        st.pyplot(fig)
        
        except Exception as e:
            import traceback
            st.error(f"Error during evaluation: {str(e)}")
            st.code(traceback.format_exc(), language="python")

# Define the Custom Dataset page
def custom_dataset():
    st.write("## Custom Dataset Evaluation")
    
    # Constants
    N_QUERY = 10
    N_EVALUATION_TASKS = 50
    
    # Sidebar controls
    st.sidebar.subheader("Model Configuration")
    N_WAY = st.sidebar.slider("N-Way (Classes)", min_value=2, max_value=20, value=5)
    N_SHOT = st.sidebar.slider("N-Shot (Examples per class)", min_value=1, max_value=10, value=5)
    
    # Select models to evaluate
    available_backbones = load_backbones()
    st.sidebar.subheader("Select Models to Evaluate")
    selected_models = {}
    for model_name in available_backbones.keys():
        if st.sidebar.checkbox(model_name, value=(model_name == "MobileNetV2")):
            selected_models[model_name] = available_backbones[model_name]
    
    # Initialize with selected models or at least one model if none selected
    if not selected_models:
        selected_models = {"MobileNetV2": available_backbones["MobileNetV2"]}
        st.sidebar.warning("At least one model must be selected. Using MobileNetV2 by default.")
    
    # Initialize MINDS class and add the selected models
    meta_eval = MINDS()
    for name, backbone in selected_models.items():
        proto_model = PrototypicalNetworks(backbone=backbone)
        meta_eval.add_model(name, proto_model)
    
    # Upload custom dataset
    st.write("### Upload your dataset as a ZIP file")
    st.markdown("""
    The ZIP file should contain folders, each representing a class, with images inside:
    ```
    dataset.zip
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2/
    │   ├── image1.jpg
    │   └── ...
    └── ...
    ```
    """)
    
    uploaded_file = st.file_uploader("Choose a ZIP file", type="zip")
    
    if uploaded_file is not None:
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
            
            # Load the dataset
            progress_bar = st.progress(0)
            
            try:
                # Load custom dataset
                test_set = meta_eval.load_custom_dataset(extract_dir)
                progress_bar.progress(30)
                
                # Create dataloader
                test_loader = load_test_data(
                    test_set=test_set,
                    n_way=N_WAY,
                    n_shot=N_SHOT,
                    n_query=N_QUERY,
                    n_tasks=N_EVALUATION_TASKS,
                    data_type='custom',
                    progress_bar=progress_bar
                )
                
                progress_bar.progress(100)
                st.session_state.test_set = test_set
                st.session_state.test_loader = test_loader
                st.success("Custom Dataset Loaded Successfully!")
                
                # Display dataset information
                class_names = list(test_set.class_to_idx.keys())
                st.write(f"Found {len(class_names)} classes with a total of {len(test_set)} images")
                st.write("Classes:", ", ".join(class_names[:10]) + ("..." if len(class_names) > 10 else ""))
                
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
    
    # Visualize support and query sets
    if st.button("Visualize Examples") and 'test_loader' in st.session_state:
        progress_bar = st.progress(0)
        
        # Get a batch from the dataloader
        (
            example_support_images,
            example_support_labels,
            example_query_images,
            example_query_labels,
            example_class_ids,
        ) = next(iter(st.session_state.test_loader))
        
        progress_bar.progress(50)
        
        # Create visualizations ensuring proper handling of image channels
        try:
            support_plot = plot_support_query_sets(
                example_support_images, 
                example_support_labels,
                max_examples=N_SHOT
            )
            query_plot = plot_support_query_sets(
                example_query_images,
                example_query_labels,
                max_examples=N_QUERY
            )
            
            progress_bar.progress(100)
            
            # Show support and query sets
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Support Set")
                st.pyplot(support_plot)
            
            with col2:
                st.write("### Query Set")
                st.pyplot(query_plot)
        except Exception as e:
            import traceback
            st.error(f"Error visualizing examples: {str(e)}")
            st.code(traceback.format_exc(), language="python")
    
    # Evaluate models
    if st.button("Evaluate Models") and 'test_loader' in st.session_state:
        st.write("### Evaluation Results")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        selected_model_names = list(meta_eval.models.keys())
        status_text.text(f"Evaluating {len(selected_model_names)} models: {', '.join(selected_model_names)}")
        
        try:
            # Run evaluation
            all_metrics, all_inference = meta_eval.evaluate(
                st.session_state.test_loader,
                n_tasks=N_EVALUATION_TASKS,
                bar=False,
                webview_bar=progress_bar
            )
            
            progress_bar.progress(100)
            status_text.text("Evaluation Complete!")
            
            # Display results
            for model_name, metrics in all_metrics.items():
                with st.expander(f"{model_name} Results", expanded=True):
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                    with cols[1]:
                        st.metric("F1 Score", f"{metrics['f1']:.4f}")
                    with cols[2]:
                        st.metric("Inference Time", f"{all_inference[model_name]:.4f}s")
                    
                    # Show confusion matrix if available
                    if 'confusion_matrix' in metrics:
                        st.write("#### Confusion Matrix")
                        fig, ax = plt.subplots(figsize=(10, 8))
                        cax = ax.matshow(metrics['confusion_matrix'], cmap='Blues')
                        fig.colorbar(cax)
                        st.pyplot(fig)
        
        except Exception as e:
            import traceback
            st.error(f"Error during evaluation: {str(e)}")
            st.code(traceback.format_exc(), language="python")

# Sidebar navigation with enhanced styling
st.sidebar.markdown("<div style='text-align:center; padding: 10px 0; border-bottom: 1px solid #ccc;'><h3>Navigation</h3></div>", unsafe_allow_html=True)
selected_page = st.sidebar.radio("", ["Home", "Omniglot Dataset", "Custom Dataset"])

# Display the selected page
if selected_page == "Home":
    main_page()
elif selected_page == "Omniglot Dataset":
    omniglot_dataset()
elif selected_page == "Custom Dataset":
    custom_dataset()