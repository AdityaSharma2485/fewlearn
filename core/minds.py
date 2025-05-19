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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        print("MINDS framework initialized successfully")
    
    def add_model(self, name: str, model: FewShotModel) -> None:
        """
        Add a few-shot learning model to the framework.
        
        Args:
            name: Unique identifier for this model
            model: A few-shot learning model instance
        """
        if name in self.models:
            print(f"Warning: Overwriting existing model '{name}'")
        
        self.models[name] = model.to(self.device)
        print(f"Added model '{name}' successfully")
    
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
        if not os.path.exists(dataset_path):
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
        
        # Load dataset using ImageFolder
        dataset = datasets.ImageFolder(
            root=dataset_path,
            transform=transform
        )
        
        print(f"Loaded custom dataset from {dataset_path}: {len(dataset)} images, {len(dataset.classes)} classes")
        
        # Add class_names attribute for compatibility
        dataset.class_names = dataset.classes
        
        return dataset
    
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
            
            # Move data to device
            support_images = support_images.to(self.device)
            support_labels = support_labels.to(self.device)
            query_images = query_images.to(self.device)
            query_labels = query_labels.to(self.device)
            
            # Evaluate each model on this task
            for model_name in model_names:
                model = self.models[model_name]
                
                # Measure inference time
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                with torch.no_grad():
                    # Get model predictions
                    predictions = model(support_images, support_labels, query_images)
                    
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                
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