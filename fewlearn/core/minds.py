"""
MINDS: Minimal Instance Neural Data System

The core framework for few-shot learning evaluations.
"""

import torch
from torch import nn, Tensor
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any, Callable
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

# Use absolute imports for better package compatibility
from fewlearn.core.protocols import EvaluationProtocol
from fewlearn.models.base import FewShotModel
from fewlearn.evaluation.metrics import calculate_metrics


class MINDS:
    """
    Minimal Instance Neural Data System (MINDS).
    
    MINDS is the main framework for few-shot learning that handles:
    - Loading and managing pretrained backbone models
    - Dataset handling for few-shot scenarios
    - Evaluation using various few-shot learning protocols
    - Performance comparison across multiple models
    
    Attributes:
        device (torch.device): The device to run computations on (CPU or GPU)
        models (Dict[str, FewShotModel]): Dictionary of registered few-shot learning models
    """
    
    def __init__(self):
        """Initialize a new MINDS framework instance."""
        # Try to use GPU but with fallback mechanism
        try:
            if torch.cuda.is_available():
                # Check available GPU memory
                total_mem = torch.cuda.get_device_properties(0).total_memory
                reserved_mem = torch.cuda.memory_reserved(0)
                if reserved_mem / total_mem > 0.8:  # If more than 80% is reserved
                    print("Warning: GPU memory usage is high. Using CPU instead.")
                    self.device = torch.device('cpu')
                else:
                    self.device = torch.device('cuda')
                    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device('cpu')
                print("Using CPU: CUDA not available")
        except Exception as e:
            print(f"Error initializing CUDA: {e}. Falling back to CPU.")
            self.device = torch.device('cpu')
        
        # Set default device memory limit flag
        self.use_cpu_fallback = True
        self.models = {}
        print(f"MINDS framework initialized successfully on {self.device}")
    
    def add_model(self, name: str, model: FewShotModel) -> None:
        """
        Add a few-shot learning model to the framework.
        
        Args:
            name: Unique identifier for this model
            model: A few-shot learning model instance
        """
        if name in self.models:
            print(f"Warning: Overwriting existing model '{name}'")
        
        try:
            # Try to move model to the specified device
            self.models[name] = model.to(self.device)
        except RuntimeError as e:
            # Handle CUDA out-of-memory errors
            if "CUDA out of memory" in str(e) and self.use_cpu_fallback:
                print(f"\nWarning: CUDA out of memory when loading model '{name}'. Falling back to CPU.")
                # Set device to CPU for all future operations
                self.device = torch.device('cpu')
                # Move model to CPU instead
                self.models[name] = model.to(self.device)
                # Also move any previously loaded models to CPU
                for model_name, loaded_model in self.models.items():
                    if model_name != name:  # Skip the model we just added
                        self.models[model_name] = loaded_model.to(self.device)
            else:
                # Re-raise other errors
                raise
            
        print(f"Added model '{name}' successfully on {self.device}")
    
    def remove_model(self, name: str) -> None:
        """
        Remove a model from the framework.
        
        Args:
            name: The name of the model to remove
        """
        if name in self.models:
            del self.models[name]
            print(f"Removed model '{name}' successfully")
        else:
            print(f"Model '{name}' not found")
    
    def load_omniglot(self, image_size: int = 28) -> Tuple[Dataset, Dataset]:
        """
        Load the Omniglot dataset for few-shot learning.

        Args:
            image_size: Size to resize the images to

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        from torchvision.datasets import Omniglot
        
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),  # Convert grayscale to RGB
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet normalization for pretrained models
        ])
        
        # Load datasets
        background_set = Omniglot(
            root='./data', 
            background=True,
            download=True, 
            transform=transform
        )
        
        evaluation_set = Omniglot(
            root='./data', 
            background=False,
            download=True, 
            transform=transform
        )
        
        print(f"Loaded Omniglot dataset: {len(background_set)} background images, {len(evaluation_set)} evaluation images")
        return background_set, evaluation_set
    
    def load_custom_dataset(self, dataset_path: str) -> Dataset:
        """
        Load a custom dataset from a directory.
        
        The directory should have subdirectories for each class,
        with image files inside each class directory.

        Args:
            dataset_path: Path to the dataset directory

        Returns:
            Dataset object
        """
        from pathlib import Path
        
        # Convert to Path object for better path handling
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        try:
            # Load dataset using ImageFolder
            dataset = datasets.ImageFolder(
                root=str(dataset_path),
                transform=transform
            )
            
            print(f"Loaded custom dataset from {dataset_path}: {len(dataset)} images, {len(dataset.classes)} classes")
            
            # Add class_names attribute for compatibility
            dataset.class_names = dataset.classes
            
            return dataset
        except Exception as e:
            print(f"Error loading dataset from {dataset_path}: {str(e)}")
            # Implement a fallback custom dataset directly here
            try:
                # Create a custom dataset class inline
                class FallbackCustomDataset(Dataset):
                    def __init__(self, root_dir, transform=None):
                        self.root_dir = Path(root_dir)
                        self.transform = transform
                        self.classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
                        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
                        
                        self.samples = []
                        self.labels = []
                        self.targets = []  # Add targets for compatibility with ImageFolder
                        
                        for class_name in self.classes:
                            class_dir = self.root_dir / class_name
                            class_idx = self.class_to_idx[class_name]
                            
                            for img_name in os.listdir(class_dir):
                                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    img_path = class_dir / img_name
                                    # Store absolute path to avoid path resolution issues
                                    self.samples.append(str(img_path.absolute()))
                                    self.labels.append(class_idx)
                                    self.targets.append(class_idx)  # Store targets same as labels
                
                    def __len__(self):
                        return len(self.samples)
                
                    def __getitem__(self, idx):
                        img_path = self.samples[idx]
                        try:
                            image = Image.open(img_path).convert('RGB')
                        except Exception as e:
                            print(f"Error opening image {img_path}: {e}")
                            # Return a blank image as fallback
                            image = Image.new('RGB', (224, 224), color='gray')
                        
                        label = self.labels[idx]
                        
                        if self.transform:
                            image = self.transform(image)
                        
                        return image, label
                
                # Use our fallback implementation
                dataset = FallbackCustomDataset(
                    root_dir=dataset_path,
                    transform=transform
                )
                print(f"Loaded dataset with FallbackCustomDataset from {dataset_path}: {len(dataset)} images")
                # Add class_names attribute for compatibility
                dataset.class_names = dataset.classes
                return dataset
            except Exception as ce:
                print(f"Error loading with FallbackCustomDataset: {str(ce)}")
                # Re-raise original error if custom loader also fails
                raise e
    
    def evaluate(self, 
                 data_loader: torch.utils.data.DataLoader,
                 n_tasks: int = 100,
                 bar: bool = True,
                 webview_bar = None
                ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
        """
        Evaluate one or more models using episodic few-shot tasks.
        
        Args:
            data_loader: DataLoader providing evaluation tasks
            n_tasks: Number of tasks to evaluate on
            bar: Whether to display a progress bar in console
            webview_bar: Optional Streamlit progress bar
            
        Returns:
            Tuple of (metrics, inference_times)
        """
        if not self.models:
            raise ValueError("No models available for evaluation. Add models using add_model() method.")
        
        # Set all models to evaluation mode
        for model in self.models.values():
            model.eval()
        
        # Initialize results dictionaries
        all_metrics = {}
        all_inference_times = {}
        
        # Get model names
        model_names = list(self.models.keys())
        
        # Try using GPU first, with fallback to CPU if memory issues occur
        original_device = self.device
        memory_error_occurred = False
        
        # Iterate through tasks
        task_counter = 0
        for task_data in data_loader:
            if task_counter >= n_tasks:
                break
                
            # Update progress
            if webview_bar is not None:
                webview_bar.progress(task_counter / n_tasks)
            elif bar:
                print(f"\rEvaluating task {task_counter+1}/{n_tasks}", end="")
            
            # Extract task data
            support_images, support_labels, query_images, query_labels, _ = task_data
            
            # Handle memory errors and device transfer
            try:
                if memory_error_occurred and self.use_cpu_fallback:
                    # Already had memory error, use CPU
                    current_device = torch.device('cpu')
                else:
                    current_device = self.device
                    
                # Move data to device with proper error handling
                support_images = support_images.to(current_device)
                support_labels = support_labels.to(current_device)
                query_images = query_images.to(current_device)
                query_labels = query_labels.to(current_device)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and self.use_cpu_fallback:
                    print("\nCUDA out of memory error. Switching to CPU...")
                    memory_error_occurred = True
                    current_device = torch.device('cpu')
                    
                    # Try again with CPU
                    support_images = support_images.to(current_device)
                    support_labels = support_labels.to(current_device)
                    query_images = query_images.to(current_device)
                    query_labels = query_labels.to(current_device)
                    
                    # Also move models to CPU
                    for model_name in model_names:
                        self.models[model_name] = self.models[model_name].to(current_device)
                else:
                    raise  # Re-raise if not an out of memory error or if fallback is disabled
            
            # Evaluate each model on this task
            for model_name in model_names:
                model = self.models[model_name]
                # Ensure model is on the same device as data
                if next(model.parameters()).device != current_device:
                    model = model.to(current_device)
                
                # Measure inference time - handle different devices
                if current_device.type == 'cuda':
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    
                    start_time.record()
                    with torch.no_grad():
                        # Get model predictions
                        predictions = model(support_images, support_labels, query_images)
                        
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                else:
                    # For CPU, use simple time measurement
                    import time
                    start_time = time.time()
                    with torch.no_grad():
                        # Get model predictions
                        predictions = model(support_images, support_labels, query_images)
                    inference_time = time.time() - start_time
                
                # Convert predictions and labels to numpy arrays
                predictions_np = predictions.cpu().numpy()
                query_labels_np = query_labels.cpu().numpy()
                
                # Get unique labels from support set for mapping predictions
                unique_labels = torch.unique(support_labels).cpu().numpy()
                
                # Convert prediction scores to class indices
                if len(predictions_np.shape) > 1 and predictions_np.shape[1] > 1:
                    # Get indices of max scores
                    pred_indices = np.argmax(predictions_np, axis=1)
                    # Map indices back to original class labels if needed
                    if len(unique_labels) > 0:
                        predictions_mapped = np.array([unique_labels[idx] if idx < len(unique_labels) else idx for idx in pred_indices])
                    else:
                        predictions_mapped = pred_indices
                else:
                    # Predictions are already indices
                    predictions_mapped = predictions_np
                
                # Calculate metrics for this task
                task_metrics = calculate_metrics(predictions_mapped, query_labels_np)
                
                # Accumulate metrics
                if model_name not in all_metrics:
                    all_metrics[model_name] = task_metrics
                    all_inference_times[model_name] = inference_time
                else:
                    for key, value in task_metrics.items():
                        if key == 'confusion_matrix':
                            # For confusion matrix, sum the matrices
                            if key in all_metrics[model_name]:
                                all_metrics[model_name][key] += value
                            else:
                                all_metrics[model_name][key] = value
                        else:
                            # For other metrics, accumulate as usual
                            if key in all_metrics[model_name]:
                                all_metrics[model_name][key] += value
                            else:
                                all_metrics[model_name][key] = value
                    
                    all_inference_times[model_name] += inference_time
            
            task_counter += 1
        
        # Calculate averages
        for model_name in model_names:
            for key in all_metrics[model_name]:
                # Special handling for confusion matrix which should not be averaged
                if key == 'confusion_matrix' and isinstance(all_metrics[model_name][key], np.ndarray):
                    continue
                # Convert all other metrics to float before division
                elif isinstance(all_metrics[model_name][key], (int, np.integer, float, np.floating)):
                    all_metrics[model_name][key] = float(all_metrics[model_name][key]) / task_counter
                # Handle other cases (most likely already float arrays)
                else:
                    try:
                        all_metrics[model_name][key] /= task_counter
                    except Exception as e:
                        print(f"Warning: Could not average metric '{key}': {str(e)}")
                
            all_inference_times[model_name] /= task_counter
            
        # Print final newline if using console progress
        if bar:
            print("")
            
        # Complete progress bar
        if webview_bar is not None:
            webview_bar.progress(1.0)
            
        return all_metrics, all_inference_times
    
    def get_best_model(self, 
                       results: Dict[str, Dict[str, Any]], 
                       metric: str = "accuracy"
                      ) -> Tuple[str, FewShotModel]:
        """
        Get the best performing model from evaluation results.
        
        Args:
            results: Evaluation results from evaluate()
            metric: The metric to use for ranking models
            
        Returns:
            Tuple of (model_name, model_instance)
        """
        if not results:
            raise ValueError("No results provided")
            
        # Find the model with the best metric value
        model_scores = [(name, float(data["metrics"][metric])) 
                        for name, data in results.items()
                        if metric in data["metrics"]]
        
        if not model_scores:
            raise ValueError(f"Metric '{metric}' not found in results")
            
        best_model_name, _ = max(model_scores, key=lambda x: x[1])
        
        return best_model_name, self.models[best_model_name]
    
    def export_model(self, 
                     model_name: str, 
                     format: str = "onnx", 
                     output_path: str = None
                    ) -> str:
        """
        Export a model for deployment.
        
        Args:
            model_name: Name of the model to export
            format: Export format ("onnx", "torchscript", etc.)
            output_path: Path to save the exported model
            
        Returns:
            Path to the exported model file
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
            
        # Import export utilities on demand to avoid dependencies
        from fewlearn.utils.export import export_model
        
        return export_model(
            model=self.models[model_name],
            format=format,
            output_path=output_path or f"{model_name}.{format}"
        ) 