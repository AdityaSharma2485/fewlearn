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
import shutil
import time
import random

# Import from restructured fewlearn module
from fewlearn.core.minds import MINDS
from fewlearn.models.prototypical import PrototypicalNetworks
from fewlearn.evaluation.evaluator import Evaluator
from fewlearn.visualization import plot_support_query_sets

# Add torch multiprocessing for proper parallelization
import torch.multiprocessing as mp
# Set multiprocessing method safely
if __name__ == '__main__' and mp.get_start_method(allow_none=True) is None:
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # Method already set

# Main page heading with styling improvements
st.markdown("<h1 style='text-align: center; font-size: 5em; margin-bottom: 0.1em;'>M.I.N.D.S</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; font-size: 2.5em; margin-top: -20px;'>Minimal Instance Neural Data System</h2>", unsafe_allow_html=True)
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
        background-color: #e74c3c;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background-color: #c0392b;
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
        border-left: 3px solid #e74c3c;
    }
    </style>
""", unsafe_allow_html=True)

# Add at the beginning of the file, after imports
def create_temp_dir():
    """Create a temporary directory that persists during the session"""
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()
    return st.session_state.temp_dir

def cleanup_temp_dir():
    """Clean up the temporary directory when the session ends"""
    if 'temp_dir' in st.session_state:
        try:
            shutil.rmtree(st.session_state.temp_dir)
            del st.session_state.temp_dir
        except Exception as e:
            st.warning(f"Failed to clean up temporary directory: {e}")

# Register cleanup function to be called when the app stops
if 'cleanup_registered' not in st.session_state:
    import atexit
    atexit.register(cleanup_temp_dir)
    st.session_state.cleanup_registered = True

# Initialize session state variables
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None
if 'best_model_metrics' not in st.session_state:
    st.session_state.best_model_metrics = None
if 'test_loader' not in st.session_state:
    st.session_state.test_loader = None
if 'test_set' not in st.session_state:
    st.session_state.test_set = None

# Create backbone dictionary with proper error handling
@st.cache_resource
def load_backbones():
    """Load and cache model backbones with proper error handling."""
    try:
        available_backbones = {
            "GoogleNet": models.googlenet(weights="DEFAULT"),
            "ResNet18": models.resnet18(weights="DEFAULT"),
            "InceptionV3": models.inception_v3(weights="DEFAULT"),
            "MobileNetV2": models.mobilenet_v2(weights="DEFAULT"),
            "DenseNet121": models.densenet121(weights="DEFAULT"),
        }
        
        # Modify the fully connected layer to output features
        for name, model in available_backbones.items():
            model.eval()  # Set to evaluation mode
            with torch.no_grad():  # Disable gradient computation
                if name == "InceptionV3":
                    model.fc = nn.Flatten()
                    model.aux_logits = False  # Disable auxiliary outputs
                elif hasattr(model, 'fc'):
                    model.fc = nn.Flatten()
                elif hasattr(model, 'classifier'):
                    if isinstance(model.classifier, nn.Sequential):
                        model.classifier = nn.Flatten()
                    else:
                        model.classifier = nn.Flatten()
                
                # Move model to GPU if available
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
        
        return available_backbones
    except Exception as e:
        st.error(f"Error loading model backbones: {str(e)}")
        return {}  # Return empty dict instead of None to prevent further errors

# Helper function to update progress bars
def update_progress(progress_bar, step):
    progress_bar.progress(step)

class DatasetWrapper:
    """Wrapper class to handle dataset labels and methods."""
    def __init__(self, dataset, data_type='custom'):
        self.dataset = dataset
        self.data_type = data_type
        
        if data_type == 'omniglot':
            self.labels = [instance[1] for instance in dataset._flat_character_images]
        else:
            self.labels = [instance[1] for instance in dataset.samples]
        
    def get_labels(self):
        """Return the labels for the dataset."""
        return self.labels
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

# Helper function to load test data with proper error handling and parallelization
def load_test_data(test_set, n_way, n_shot, n_query, n_tasks, progress_bar, data_type='custom'):
    """
    Load test data with proper error handling and parallelization.
    
    Args:
        test_set: Dataset to sample from
        n_way: Number of classes per class
        n_shot: Number of support examples per class
        n_query: Number of query examples per class
        n_tasks: Number of tasks to generate
        progress_bar: Streamlit progress bar
        data_type: Type of dataset ('omniglot' or 'custom')
    
    Returns:
        DataLoader: Configured data loader for few-shot tasks
    """
    try:
        # Validate input parameters
        if test_set is None:
            raise ValueError("Test set is None")
        
        if n_way <= 0 or n_shot <= 0 or n_query <= 0 or n_tasks <= 0:
            raise ValueError(f"Invalid parameters: n_way={n_way}, n_shot={n_shot}, n_query={n_query}, n_tasks={n_tasks}")
        
        # Count available classes and validate against n_way
        unique_labels = set()
        if data_type == 'omniglot':
            unique_labels = set([instance[1] for instance in test_set._flat_character_images])
        else:
            unique_labels = set([instance[1] for instance in test_set.samples])
        
        if len(unique_labels) < n_way:
            raise ValueError(f"Dataset has only {len(unique_labels)} classes but n_way={n_way} was requested")
        
        # Wrap the dataset to handle labels properly
        wrapped_dataset = DatasetWrapper(test_set, data_type)
        
        update_progress(progress_bar, 20)
        
        # Determine optimal number of workers based on system
        num_workers = 0  # Set to 0 for Windows compatibility
    
        test_sampler = TaskSampler(
            wrapped_dataset,
            n_way=n_way,
            n_shot=n_shot,
            n_query=n_query,
            n_tasks=n_tasks
        )

        test_loader = DataLoader(
            wrapped_dataset,
            batch_sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=test_sampler.episodic_collate_fn,
        )
        
        update_progress(progress_bar, 60)
        
        return test_loader
    except Exception as e:
        st.error(f"Error creating data loader: {str(e)}")
        if "CUDA out of memory" in str(e):
            st.warning("CUDA ran out of memory. Try reducing the number of tasks, shots, or query images.")
        elif "n_way" in str(e):
            st.warning("The dataset doesn't have enough classes for the requested few-shot setting.")
        return None

def show_predictions():
    if 'best_model' not in st.session_state or 'test_loader' not in st.session_state:
        st.error("Please run evaluation first!")
        return
    
    try:
        # Create a popup window using st.dialog
        prediction_container = st.container()
        with prediction_container:
            st.write(f"### Predictions using {st.session_state.best_model_name}")
            st.write(f"Model Accuracy: {st.session_state.best_model_metrics['accuracy']:.2%}")
            
            # Get predictions for a new batch
            progress_bar = st.progress(0)
            st.text("Getting predictions...")
            
            # Get a batch from the dataloader
            (
                example_support_images,
                example_support_labels,
                example_query_images,
                example_query_labels,
                example_class_ids,
            ) = next(iter(st.session_state.test_loader))
            
            progress_bar.progress(30)
            
            # Move tensors to appropriate device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            example_support_images = example_support_images.to(device)
            example_support_labels = example_support_labels.to(device)
            example_query_images = example_query_images.to(device)
            example_query_labels = example_query_labels.to(device)
            
            # Get predictions from the best model
            with torch.no_grad():
                predictions = st.session_state.best_model(
                    example_support_images,
                    example_support_labels,
                    example_query_images
                )
                predicted_labels = torch.argmax(predictions, dim=1)
            
            progress_bar.progress(60)
            
            # Display results in a grid
            st.write("#### Sample Predictions")
            
            # Convert tensors to numpy for visualization
            query_images = example_query_images.cpu().numpy()
            actual_labels = example_query_labels.cpu().numpy()
            predicted_labels = predicted_labels.cpu().numpy()
            
            # Map class indices to class names - handle both Omniglot and custom datasets
            dataset_type = getattr(st.session_state.test_set, '__class__', None).__name__

            # Convert between the task-specific labels and the original character labels
            is_omniglot = dataset_type == 'Omniglot'
            
            # Create class mappings
            if is_omniglot and hasattr(st.session_state.test_set, '_characters'):
                # For Omniglot, use the mapping between task indices (0-n_way) and original character indices
                # We need to map the task-specific labels to the original character indices
                unique_labels = torch.unique(example_support_labels).cpu().numpy()
                # If we have class_ids, they correspond to the original character indices
                if isinstance(example_class_ids, list) and len(example_class_ids) >= len(unique_labels):
                    # Class IDs in episodic sampling map class indices in the task to original dataset indices
                    idx_to_class = {}
                    # For each task-specific label (0 to n_way-1), map to original character label
                    for i, label in enumerate(unique_labels):
                        original_idx = example_class_ids[i]  # This is the original character index
                        # Verify if this is an index or a path
                        if isinstance(original_idx, int):
                            # Use the _characters mapping at the original index
                            if original_idx < len(st.session_state.test_set._characters):
                                idx_to_class[int(label)] = st.session_state.test_set._characters[original_idx]
                            else:
                                idx_to_class[int(label)] = f"Character {original_idx}"
                        else:
                            # For paths or strings, extract the meaningful part
                            if isinstance(original_idx, str) and '/' in original_idx:
                                # Example: data/omniglot/images_evaluation/Grantha/character12/0861_18.png
                                parts = original_idx.split('/')
                                if len(parts) >= 3:
                                    alphabet = parts[-3]  # e.g., 'Grantha'
                                    char_part = parts[-2]  # e.g., 'character12'
                                    char_num = char_part.replace('character', '')
                                    idx_to_class[int(label)] = f"{alphabet} {char_num}"
                                else:
                                    idx_to_class[int(label)] = f"Character {i}"
                            else:
                                idx_to_class[int(label)] = f"Character {i}"
                else:
                    # Fallback if we can't map using class_ids
                    idx_to_class = {int(i): f"Class {i}" for i in range(len(unique_labels))}
            elif hasattr(st.session_state.test_set, 'class_to_idx'):
                # Custom dataset with class_to_idx mapping
                class_to_idx = st.session_state.test_set.class_to_idx
                idx_to_class = {v: k for k, v in class_to_idx.items()}
            else:
                # Fallback option
                idx_to_class = {i: f"Class {i}" for i in range(100)}  # Arbitrary limit
            
            # Create a grid of predictions
            num_samples = min(10, len(query_images))  # Show up to 10 samples
            
            # Shuffle and select diverse examples to ensure variety
            # First get indices grouped by class
            class_indices = {}
            for i in range(len(actual_labels)):
                label = int(actual_labels[i])
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(i)
            
            # Create a diverse selection by selecting samples from different classes
            selected_indices = []
            available_classes = list(class_indices.keys())
            
            # First prioritize having at least one example of each available class
            random.shuffle(available_classes)
            for cls in available_classes:
                if class_indices[cls]:
                    selected_idx = random.choice(class_indices[cls])
                    selected_indices.append(selected_idx)
                    class_indices[cls].remove(selected_idx)
                    
                    # Stop if we've reached our sample limit
                    if len(selected_indices) >= num_samples:
                        break
            
            # If we still need more, fill in with random examples
            while len(selected_indices) < num_samples and any(cls for cls in class_indices.values() if cls):
                # Get classes that still have examples
                remaining_classes = [cls for cls in class_indices if class_indices[cls]]
                if not remaining_classes:
                    break
                    
                cls = random.choice(remaining_classes)
                selected_idx = random.choice(class_indices[cls])
                selected_indices.append(selected_idx)
                class_indices[cls].remove(selected_idx)
            
            # Shuffle the final selection
            random.shuffle(selected_indices)
            
            # Display the selected examples
            cols = st.columns(2)
            for i, idx in enumerate(selected_indices):
                with cols[i % 2]:
                    # Display the image
                    img = query_images[idx].transpose(1, 2, 0)
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # Added epsilon to prevent division by zero
                    st.image(img, caption=f"Sample {i + 1}", use_column_width=True)
                    
                    # Show actual and predicted labels
                    label_idx = int(actual_labels[idx])
                    pred_idx = int(predicted_labels[idx])
                    
                    # Get the class names using our mapping
                    actual = idx_to_class.get(label_idx, f"Class {label_idx}")
                    predicted = idx_to_class.get(pred_idx, f"Class {pred_idx}")
                    is_correct = label_idx == pred_idx
                    
                    st.markdown(
                        f"**Actual:** {actual}\n\n"
                        f"**Predicted:** {predicted}\n\n"
                        f"**Status:** {'✅ Correct' if is_correct else '❌ Wrong'}"
                    )
                    st.markdown("---")
            
            progress_bar.progress(100)
            st.success("Predictions complete!")
            
    except Exception as e:
        st.error(f"Error showing predictions: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

# Define the Omniglot dataset page
def omniglot_dataset():
    st.write("## Omniglot Dataset Evaluation")
    
    # Constants
    N_QUERY = 10
    N_EVALUATION_TASKS = 100
    
    # Input boxes with improved layout
    contain_class, contain_shots = st.columns(2)

    with contain_class:
        N_WAY = st.number_input("Number of Classes:", min_value=1, value=5, step=1)

    with contain_shots:
        N_SHOT = st.number_input("Number of Shots:", min_value=1, value=5, step=1)
    
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
    
    # Initialize session state variables if they don't exist
    if 'test_loader' not in st.session_state:
        st.session_state.test_loader = None

    # Define three buttons in a row with improved styling
    left, middle, right = st.columns(3)
    
    # Button 1: Load OMNIGLOT dataset
    if left.button('LOAD OMNIGLOT', use_container_width=True):
        try:
            progress_bar = st.progress(0)
            _, test_set = meta_eval.load_omniglot(image_size=128)
            progress_bar.progress(30)
            
            # Add _characters attribute to the Omniglot dataset for displaying character names
            # This is necessary for compatibility with the reference implementation
            if not hasattr(test_set, '_characters'):
                # Create a mapping from label indices to readable character names
                all_chars = []
                for path, label in test_set._flat_character_images:
                    # Extract alphabet and character number from path
                    parts = path.split('/')
                    if len(parts) >= 3:
                        alphabet = parts[-3]  # e.g., 'Japanese'
                        char_num = parts[-2]  # e.g., 'character01'
                        char_name = f"{alphabet} {char_num.replace('character', '')}"
                        # Ensure we have enough elements in the list
                        while len(all_chars) <= label:
                            all_chars.append(None)
                        all_chars[label] = char_name
                    else:
                        all_chars.append(f"Character {label}")
                
                # Assign the mapping to the dataset
                test_set._characters = all_chars
                print(f"Added _characters attribute with {len(all_chars)} character names")
            
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
    
    # Button 2: Visualize
    if middle.button("VISUALIZE", use_container_width=True):
        if 'test_loader' not in st.session_state or st.session_state.test_loader is None:
            st.error("Please load the dataset first!")
            return
            
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
            # Use the original plot_support_query_sets function
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
            st.error(f"Error visualizing examples: {str(e)}")

    # Button 3: Evaluate
    if right.button("EVALUATE", use_container_width=True):
        if 'test_loader' not in st.session_state or st.session_state.test_loader is None:
            st.error("Please load the dataset first!")
            return
            
        st.write("### Evaluation Results")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            selected_model_names = list(meta_eval.models.keys())
            status_text.text(f"Evaluating {len(selected_model_names)} models: {', '.join(selected_model_names)}")
            
            # Start timing
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.time()
            
            # Evaluate all models in one call - MINDS.evaluate internally handles all models
            # The evaluate method returns a tuple of (metrics, inference_times)
            all_metrics, all_inference = meta_eval.evaluate(
                data_loader=st.session_state.test_loader,
                n_tasks=N_EVALUATION_TASKS,
                bar=False,
                webview_bar=progress_bar
            )
            
            # End timing
            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
                total_time = start_event.elapsed_time(end_event) / 1000  # Convert ms to seconds
            else:
                total_time = time.time() - start_time
            
            if not all_metrics:
                raise ValueError("No evaluation results returned")
            
            progress_bar.progress(100)
            status_text.text("Evaluation Complete!")
            
            # Find the best model based on accuracy
            best_model = max(all_metrics.items(), key=lambda x: x[1]['accuracy'])
            st.session_state.best_model_name = best_model[0]
            st.session_state.best_model_metrics = best_model[1]
            st.session_state.best_model = meta_eval.models[best_model[0]]
            
            # Display results in a cleaner format
            st.write("#### Model Performance Summary")
            
            # Display individual model results
            for model_name, metrics in all_metrics.items():
                with st.container():
                    is_best = model_name == st.session_state.best_model_name
                    title = f"**{model_name} Detailed Results**"
                    if is_best:
                        title += " 🏆"
                    st.markdown(title)
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                    with cols[1]:
                        st.metric("F1 Score", f"{metrics['f1']:.4f}")
                    with cols[2]:
                        total_time = all_inference[model_name]
                        st.metric("Total Time", f"{total_time:.3f}s")
                    
                    # Calculate per-task timing
                    avg_time = total_time / N_EVALUATION_TASKS
                    tasks_per_second = N_EVALUATION_TASKS / total_time if total_time > 0 else 0
                    
                    # Format time with appropriate units
                    if avg_time < 0.001:
                        time_str = f"{avg_time * 1e6:.2f}μs"
                    elif avg_time < 1:
                        time_str = f"{avg_time * 1e3:.2f}ms"
                    else:
                        time_str = f"{avg_time:.3f}s"
                    
                    st.markdown(f"*Average time per task: {time_str}*")
                    st.markdown(f"*Processing speed: {tasks_per_second:.1f} tasks/s*")
                    st.markdown("---")

            # Automatically show predictions
            show_predictions()
        
        except Exception as e:
            st.error(f"Error during evaluation: {str(e)}")
            st.info("If you're seeing CUDA out of memory errors, try reducing the number of evaluation tasks or using a smaller model.")

def validate_dataset_requirements(dataset_dir, n_way, n_shot, n_query):
    """
    Validate that the dataset meets all requirements for few-shot learning.
    
    Args:
        dataset_dir: Path to dataset directory
        n_way: Number of classes per task
        n_shot: Number of support examples per class
        n_query: Number of query examples per class
    
    Returns:
        tuple: (is_valid, message)
    """
    try:
        class_folders = [d for d in os.listdir(dataset_dir) 
                        if os.path.isdir(os.path.join(dataset_dir, d))]
        
        if not class_folders:
            return False, "No class folders found in the dataset"
        
        if len(class_folders) < n_way:
            return False, f"Found only {len(class_folders)} classes, but n_way={n_way} classes are required"
        
        min_required = n_shot + n_query
        class_counts = {}
        
        for class_folder in class_folders:
            class_path = os.path.join(dataset_dir, class_folder)
            n_images = len([f for f in os.listdir(class_path) 
                          if os.path.isfile(os.path.join(class_path, f)) and 
                          f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            class_counts[class_folder] = n_images
            
            if n_images < min_required:
                return False, f"Class '{class_folder}' has only {n_images} images. Need at least {min_required} images (N_SHOT + N_QUERY = {n_shot} + {n_query})"
        
        return True, class_counts
    except Exception as e:
        return False, f"Error validating dataset: {str(e)}"

# Define the Custom Dataset page
def custom_dataset():
    st.write("## Custom Dataset Evaluation")
    
    # Constants
    N_EVALUATION_TASKS = 50
    
    # Display the folder structure chart
    st.write("### Expected Dataset Structure")
    st.markdown("""
    Please prepare your dataset in one of these two formats:
    
    1. **Direct folder structure:**
    ```
    dataset_folder/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2/
    │   ├── image1.jpg
    │   └── ...
    └── ...
    ```
    
    2. **Zipped folder structure:**
    ```
    dataset.zip
    └── dataset_folder/      # Main folder containing class folders
        ├── class1/
        │   ├── image1.jpg
        │   └── ...
        ├── class2/
        │   ├── image1.jpg
        │   └── ...
        └── ...
    ```
    
    **Note:** 
    - Each class should have its own folder containing image files
    - Supported image formats: .jpg, .jpeg, .png
    - Make sure each class has enough images for the specified number of shots and queries
    """)
    
    # File uploader for zip file
    uploaded_file = st.file_uploader("Upload your dataset (ZIP file)", type="zip", 
                                   on_change=cleanup_temp_dir)  # Clean up when new file is uploaded
    
    # Input boxes with improved layout
    contain_class, contain_shots = st.columns(2)

    with contain_class:
        N_WAY = st.number_input("Number of Classes:", min_value=1, value=5, step=1)

    with contain_shots:
        N_SHOT = st.number_input("Number of Shots:", min_value=1, value=5, step=1)

    N_QUERY = st.number_input("Number of Query Images", min_value=1, value=10, step=1)
    
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
    
    # Initialize session state variables if they don't exist
    if 'test_loader' not in st.session_state:
        st.session_state.test_loader = None

    # Define three buttons in a row with improved styling
    left, middle, right = st.columns(3)
    
    # Button 1: Load Custom dataset
    if left.button('LOAD DATASET', use_container_width=True):
        if uploaded_file is None:
            st.error("Please upload a dataset first!")
            return
            
        try:
            progress_bar = st.progress(0)
            
            # Create a persistent temporary directory
            temp_dir = create_temp_dir()
            
            # Save and extract the uploaded file
            zip_path = os.path.join(temp_dir, "dataset.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            extract_dir = os.path.join(temp_dir, "extracted")
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            os.makedirs(extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            progress_bar.progress(30)
            
            # Find and validate the dataset directory
            dataset_dir = extract_dir
            extracted_contents = os.listdir(extract_dir)
            if len(extracted_contents) == 1 and os.path.isdir(os.path.join(extract_dir, extracted_contents[0])):
                potential_main_dir = os.path.join(extract_dir, extracted_contents[0])
                if any(os.path.isdir(os.path.join(potential_main_dir, d)) for d in os.listdir(potential_main_dir)):
                    dataset_dir = potential_main_dir
            
            # Validate dataset requirements
            is_valid, result = validate_dataset_requirements(dataset_dir, N_WAY, N_SHOT, N_QUERY)
            if not is_valid:
                raise ValueError(result)
            
            class_counts = result  # If valid, result contains class counts
            
            # Store the dataset directory path in session state
            st.session_state.dataset_dir = dataset_dir
            
            # Load custom dataset with error handling
            test_set = meta_eval.load_custom_dataset(dataset_dir)
            if test_set is None:
                raise ValueError("Failed to load dataset")
            
            progress_bar.progress(60)
            
            # Create dataloader with error handling
            test_loader = load_test_data(
                test_set=test_set,
                n_way=N_WAY,
                n_shot=N_SHOT,
                n_query=N_QUERY,
                n_tasks=N_EVALUATION_TASKS,
                data_type='custom',
                progress_bar=progress_bar
            )
            
            if test_loader is None:
                raise ValueError("Failed to create data loader")
            
            st.session_state.test_loader = test_loader
            st.session_state.test_set = test_set
                
            progress_bar.progress(100)
            st.success("Custom Dataset Loaded Successfully!")
            
            # Display dataset information
            class_names = list(test_set.class_to_idx.keys())
            st.write(f"Found {len(class_names)} classes with a total of {len(test_set)} images")
            st.write("Classes:", ", ".join(class_names[:10]) + ("..." if len(class_names) > 10 else ""))
        
            # Display sample counts per class
            st.write("### Samples per Class")
            
            # Create a bar chart of sample counts
            if len(class_counts) > 0:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(class_counts.keys(), class_counts.values())
                plt.xticks(rotation=45, ha='right')
                plt.xlabel('Classes')
                plt.ylabel('Number of Samples')
                plt.title('Sample Distribution Across Classes')
                plt.tight_layout()
                st.pyplot(fig)
            
        except Exception as e:
            cleanup_temp_dir()
            st.error(f"Error loading dataset: {str(e)}")
            if "No class folders found" in str(e):
                st.info("Please make sure your ZIP file contains a folder structure with class folders as shown above.")
            elif "has only" in str(e):
                st.info("Please ensure each class has enough images for the specified number of shots and queries.")

    # Button 2: Visualize
    if middle.button("VISUALIZE", use_container_width=True):
        if 'test_loader' not in st.session_state or st.session_state.test_loader is None:
            st.error("Please load the dataset first!")
            return
            
        try:
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
                # Use the original plot_support_query_sets function
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
                st.error(f"Error visualizing examples: {str(e)}")
                
        except Exception as e:
            st.error(f"Error visualizing examples: {str(e)}")
            st.info("If you're seeing memory errors, try reducing the number of shots or query images.")

    # Button 3: Evaluate
    if right.button("EVALUATE", use_container_width=True):
        if 'test_loader' not in st.session_state or st.session_state.test_loader is None:
            st.error("Please load the dataset first!")
            return
            
        st.write("### Evaluation Results")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            selected_model_names = list(meta_eval.models.keys())
            status_text.text(f"Evaluating {len(selected_model_names)} models: {', '.join(selected_model_names)}")
            
            # Start timing
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.time()
            
            # Evaluate all models in one call - MINDS.evaluate internally handles all models
            # The evaluate method returns a tuple of (metrics, inference_times)
            all_metrics, all_inference = meta_eval.evaluate(
                data_loader=st.session_state.test_loader,
                n_tasks=N_EVALUATION_TASKS,
                bar=False,
                webview_bar=progress_bar
            )
            
            # End timing
            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
                total_time = start_event.elapsed_time(end_event) / 1000  # Convert ms to seconds
            else:
                total_time = time.time() - start_time
            
            if not all_metrics:
                raise ValueError("No evaluation results returned")
            
            progress_bar.progress(100)
            status_text.text("Evaluation Complete!")
            
            # Find the best model based on accuracy
            best_model = max(all_metrics.items(), key=lambda x: x[1]['accuracy'])
            st.session_state.best_model_name = best_model[0]
            st.session_state.best_model_metrics = best_model[1]
            st.session_state.best_model = meta_eval.models[best_model[0]]
            
            # Display results in a cleaner format
            st.write("#### Model Performance Summary")
            
            # Display individual model results
            for model_name, metrics in all_metrics.items():
                with st.container():
                    is_best = model_name == st.session_state.best_model_name
                    title = f"**{model_name} Detailed Results**"
                    if is_best:
                        title += " 🏆"
                    st.markdown(title)
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                    with cols[1]:
                        st.metric("F1 Score", f"{metrics['f1']:.4f}")
                    with cols[2]:
                        total_time = all_inference[model_name]
                        st.metric("Total Time", f"{total_time:.3f}s")
                    
                    # Calculate per-task timing
                    avg_time = total_time / N_EVALUATION_TASKS
                    tasks_per_second = N_EVALUATION_TASKS / total_time if total_time > 0 else 0
                    
                    # Format time with appropriate units
                    if avg_time < 0.001:
                        time_str = f"{avg_time * 1e6:.2f}μs"
                    elif avg_time < 1:
                        time_str = f"{avg_time * 1e3:.2f}ms"
                    else:
                        time_str = f"{avg_time:.3f}s"
                    
                    st.markdown(f"*Average time per task: {time_str}*")
                    st.markdown(f"*Processing speed: {tasks_per_second:.1f} tasks/s*")
                    st.markdown("---")

            # Automatically show predictions
            show_predictions()
        
        except Exception as e:
            st.error(f"Error during evaluation: {str(e)}")
            st.info("If you're seeing CUDA out of memory errors, try reducing the number of evaluation tasks or using a smaller model.")

# Sidebar navigation with enhanced styling
st.sidebar.markdown("<div style='text-align:center; padding: 10px 0; border-bottom: 1px solid #ccc;'><h3>Navigation</h3></div>", unsafe_allow_html=True)
selected_page = st.sidebar.selectbox("", ["Home", "Omniglot Dataset", "Custom Dataset"])

# Display the selected page
if selected_page == "Home":
    st.write("## About Few-Shot Learning")

    import textwrap
    html_home_intro = textwrap.dedent("""\
<div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h3 style='color: #e74c3c;'>What is Few-Shot Learning?</h3>
    <p>Few-Shot Learning is a machine learning approach designed to learn from limited examples, much like how humans can recognize new concepts from just a few instances. It's particularly valuable when:</p>
    <ul>
        <li>Labeled data is scarce or expensive to obtain</li>
        <li>New classes emerge that weren't in the original training data</li>
        <li>Systems need to adapt quickly to new scenarios with minimal examples</li>
    </ul>
</div>

<div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h3 style='color: #e74c3c;'>How the FewLearn Library Works</h3>
    <p>FewLearn is a powerful, flexible framework for few-shot learning experiments that uses Prototypical Networks at its core. The workflow consists of several key steps:</p>
    <ol>
        <li><strong>Data Preparation</strong>: Images are organized into support sets (examples for learning) and query sets (images to classify).</li>
        <li><strong>Feature Extraction</strong>: A backbone network (like ResNet, MobileNet, etc.) extracts meaningful features from images.</li>
        <li><strong>Prototype Calculation</strong>: For each class in the support set, a "prototype" (average representation) is computed.</li>
        <li><strong>Distance Measurement</strong>: Query images are classified based on their distance to each class prototype.</li>
        <li><strong>Model Evaluation</strong>: Performance is measured across multiple random tasks to ensure robustness.</li>
    </ol>
</div>

<div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h3 style='color: #e74c3c;'>Using this Demo App</h3>
    <p>This demonstration app lets you explore few-shot learning in action:</p>
    <ul>
        <li><strong>Omniglot Dataset</strong>: Experiment with a standard few-shot learning benchmark containing handwritten characters from different alphabets.</li>
        <li><strong>Custom Dataset</strong>: Upload your own images to test few-shot learning on your specific use case.</li>
    </ul>
    <p>In both modes, you can:</p>
    <ul>
        <li>Select different backbone models to compare their performance</li>
        <li>Adjust the number of classes (N-way) and examples per class (K-shot)</li>
        <li>Visualize support and query sets to understand the learning task</li>
        <li>Evaluate models and see detailed performance metrics</li>
        <li>View model predictions with confidence scores</li>
    </ul>
</div>
""")
    st.markdown(html_home_intro, unsafe_allow_html=True)

    # Display flowchart title
    st.markdown("<h3 style='color: #e74c3c; text-align: center;'>How Prototypical Networks Work</h3>", unsafe_allow_html=True)
    
    # Display flowchart using a simple image
    from PIL import Image
    import io
    
    # Define a flowchart placeholder - this should be replaced with the actual image file
    # You can create a file uploader here if the image should be uploaded by users
    flowchart_path = "flowchart.png"  # Change this to the actual path of your image
    
    try:
        flowchart_image = Image.open(flowchart_path)
        st.image(flowchart_image, caption="Prototypical Networks Workflow", use_column_width=True)
    except (FileNotFoundError, IOError):
        # If the image doesn't exist, show a message about the workflow
        st.error("Flowchart image not found. Please upload an image named 'flowchart.png' to visualize the workflow.")
        
        # Create a text description of the workflow as fallback
        st.markdown("""
        **Prototypical Networks Workflow:**
        1. Query images are processed through the backbone network to extract features
        2. Support images are similarly processed to extract features
        3. Class prototypes are calculated by averaging feature vectors for each class in the support set
        4. Distances between query features and class prototypes are calculated
        5. Query images are classified based on the nearest prototype
        6. The model is evaluated on multiple tasks to measure performance
        """)
    
    # Caption for the flowchart
    st.markdown("<p style='text-align: center; font-style: italic; margin-top: 20px;'>The model learns to classify by computing distances between query images and class prototypes in the embedded feature space.</p>", unsafe_allow_html=True)
    
    # Getting started section
    getting_started_html = textwrap.dedent("""\
<div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-top: 30px;'>
    <h3 style='color: #e74c3c;'>Getting Started</h3>
    <p>Select a dataset option from the sidebar to begin exploring few-shot learning:</p>
    <ul>
        <li><strong>Omniglot Dataset</strong>: Experiment with handwritten characters from different alphabets</li>
        <li><strong>Custom Dataset</strong>: Upload your own images organized in class folders</li>
    </ul>
</div>
""")
    st.markdown(getting_started_html, unsafe_allow_html=True)
elif selected_page == "Omniglot Dataset":
    omniglot_dataset()
elif selected_page == "Custom Dataset":
    custom_dataset()