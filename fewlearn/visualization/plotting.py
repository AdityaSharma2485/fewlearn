"""
Plotting functions for few-shot learning visualization.

This module provides functions for visualizing the results of
few-shot learning evaluations, including plots of embeddings,
confusion matrices, and performance comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from typing import Dict, List, Union, Optional, Callable, Any, Tuple
import torch
from torch import Tensor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd


def plot_prototype_embeddings(
    support_embeddings: np.ndarray,
    support_labels: np.ndarray,
    query_embeddings: Optional[np.ndarray] = None,
    query_labels: Optional[np.ndarray] = None,
    predicted_labels: Optional[np.ndarray] = None,
    method: str = 'pca',
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.7,
    title: Optional[str] = None,
    show_prototypes: bool = True
) -> Figure:
    """
    Plot embeddings from a few-shot learning model.
    
    Args:
        support_embeddings: Embeddings of support examples
        support_labels: Labels of support examples
        query_embeddings: Embeddings of query examples
        query_labels: True labels of query examples
        predicted_labels: Predicted labels of query examples
        method: Dimensionality reduction method ('pca' or 'tsne')
        figsize: Figure size
        alpha: Alpha value for scatter plots
        title: Figure title
        show_prototypes: Whether to show class prototypes
        
    Returns:
        Matplotlib figure
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(support_embeddings, Tensor):
        support_embeddings = support_embeddings.detach().cpu().numpy()
    if isinstance(support_labels, Tensor):
        support_labels = support_labels.detach().cpu().numpy()
    if isinstance(query_embeddings, Tensor) and query_embeddings is not None:
        query_embeddings = query_embeddings.detach().cpu().numpy()
    if isinstance(query_labels, Tensor) and query_labels is not None:
        query_labels = query_labels.detach().cpu().numpy()
    if isinstance(predicted_labels, Tensor) and predicted_labels is not None:
        predicted_labels = predicted_labels.detach().cpu().numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Combine embeddings for dimensionality reduction
    all_embeddings = support_embeddings
    if query_embeddings is not None:
        all_embeddings = np.vstack([all_embeddings, query_embeddings])
    
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:  # default to PCA
        reducer = PCA(n_components=2)
        
    reduced_embeddings = reducer.fit_transform(all_embeddings)
    
    # Split reduced embeddings back into support and query
    support_reduced = reduced_embeddings[:len(support_embeddings)]
    query_reduced = None
    if query_embeddings is not None:
        query_reduced = reduced_embeddings[len(support_embeddings):]
    
    # Get unique classes
    unique_classes = np.unique(support_labels)
    n_way = len(unique_classes)
    
    # Plot support examples
    for i, class_idx in enumerate(unique_classes):
        mask = support_labels == class_idx
        ax.scatter(
            support_reduced[mask, 0],
            support_reduced[mask, 1],
            marker='o',
            s=100,
            label=f"Class {class_idx} (Support)",
            alpha=alpha
        )
    
    # Plot query examples if provided
    if query_reduced is not None and query_labels is not None:
        for i, class_idx in enumerate(unique_classes):
            mask = query_labels == class_idx
            if predicted_labels is not None:
                # Plot correct predictions
                correct_mask = (query_labels == predicted_labels) & mask
                if np.any(correct_mask):
                    ax.scatter(
                        query_reduced[correct_mask, 0],
                        query_reduced[correct_mask, 1],
                        marker='s',
                        s=80,
                        edgecolors='black',
                        linewidth=1,
                        alpha=alpha,
                        label=f"Class {class_idx} (Query, Correct)" if i == 0 else ""
                    )
                
                # Plot incorrect predictions
                incorrect_mask = (query_labels != predicted_labels) & mask
                if np.any(incorrect_mask):
                    ax.scatter(
                        query_reduced[incorrect_mask, 0],
                        query_reduced[incorrect_mask, 1],
                        marker='x',
                        s=80,
                        color='red',
                        alpha=alpha,
                        label=f"Class {class_idx} (Query, Incorrect)" if i == 0 else ""
                    )
            else:
                # Plot query examples without prediction information
                ax.scatter(
                    query_reduced[mask, 0],
                    query_reduced[mask, 1],
                    marker='s',
                    s=80,
                    alpha=alpha,
                    label=f"Class {class_idx} (Query)" if i == 0 else ""
                )
    
    # Plot prototypes if requested
    if show_prototypes:
        # Compute prototypes
        prototypes = []
        for class_idx in unique_classes:
            mask = support_labels == class_idx
            prototype = np.mean(support_reduced[mask], axis=0)
            prototypes.append(prototype)
            
        # Plot prototypes
        prototypes = np.array(prototypes)
        ax.scatter(
            prototypes[:, 0],
            prototypes[:, 1],
            marker='*',
            s=200,
            color='black',
            label='Prototypes'
        )
    
    # Add legend and title
    plt.legend(loc='upper right')
    if title:
        plt.title(title)
    else:
        plt.title(f"{n_way}-way Few-Shot Task ({method.upper()} Embeddings)")
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'Blues'
) -> Figure:
    """
    Plot confusion matrix for a few-shot classification task.
    
    Args:
        true_labels: True labels
        predicted_labels: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        figsize: Figure size
        cmap: Colormap
        
    Returns:
        Matplotlib figure
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(true_labels, Tensor):
        true_labels = true_labels.detach().cpu().numpy()
    if isinstance(predicted_labels, Tensor):
        predicted_labels = predicted_labels.detach().cpu().numpy()
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, ax=ax)
    
    # Generate class labels
    if class_names is None:
        n_classes = cm.shape[0]
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    # Set ticks and labels
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    # Add labels
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    # Add values to the plot
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.title('Confusion Matrix')
    return fig


def plot_performance_comparison(
    results: Dict[str, Dict[str, Any]],
    metric: str = 'accuracy',
    figsize: Tuple[int, int] = (10, 6),
    error_bars: bool = True,
    sort: bool = True
) -> Figure:
    """
    Plot performance comparison of multiple models.
    
    Args:
        results: Evaluation results from evaluator
        metric: Metric to compare
        figsize: Figure size
        error_bars: Whether to show error bars
        sort: Whether to sort models by performance
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract model names and metric values
    model_names = []
    metric_values = []
    error_values = []
    
    for model_name, result in results.items():
        if metric in result["metrics"]:
            model_names.append(model_name)
            metric_values.append(result["metrics"][metric])
            
            # Extract error values if available
            if error_bars and "aggregated_metrics" in result:
                std_key = f"{metric}_std"
                if std_key in result["aggregated_metrics"]:
                    error_values.append(result["aggregated_metrics"][std_key])
                else:
                    error_values.append(0)
            else:
                error_values.append(0)
    
    # Convert to numpy arrays
    metric_values = np.array(metric_values)
    error_values = np.array(error_values)
    
    # Sort if requested
    if sort:
        indices = np.argsort(metric_values)
        model_names = [model_names[i] for i in indices]
        metric_values = metric_values[indices]
        error_values = error_values[indices]
    
    # Create bar chart
    x = np.arange(len(model_names))
    bars = ax.bar(x, metric_values, yerr=error_values if error_bars else None, 
                 capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{metric_values[i]:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Models')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'Model Performance Comparison ({metric.capitalize()})')
    
    # Set x-ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Set y-axis limits
    ax.set_ylim(0, 1.1 * max(metric_values + error_values))
    
    plt.tight_layout()
    return fig


def plot_task_examples(
    support_images: np.ndarray,
    support_labels: np.ndarray,
    query_images: Optional[np.ndarray] = None,
    query_labels: Optional[np.ndarray] = None,
    predicted_labels: Optional[np.ndarray] = None,
    max_examples: int = 5,
    figsize: Tuple[int, int] = (15, 10),
    transpose_channels: bool = True,
    normalization: Optional[Tuple[float, float]] = None
) -> Figure:
    """
    Plot examples from a few-shot learning task.
    
    Args:
        support_images: Support images
        support_labels: Support labels
        query_images: Query images
        query_labels: True labels of query images
        predicted_labels: Predicted labels of query images
        max_examples: Maximum number of examples to show per class
        figsize: Figure size
        transpose_channels: Whether to transpose image channels (CHW -> HWC)
        normalization: Tuple of (mean, std) for normalization reversal
        
    Returns:
        Matplotlib figure
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(support_images, Tensor):
        support_images = support_images.detach().cpu().numpy()
    if isinstance(support_labels, Tensor):
        support_labels = support_labels.detach().cpu().numpy()
    if isinstance(query_images, Tensor) and query_images is not None:
        query_images = query_images.detach().cpu().numpy()
    if isinstance(query_labels, Tensor) and query_labels is not None:
        query_labels = query_labels.detach().cpu().numpy()
    if isinstance(predicted_labels, Tensor) and predicted_labels is not None:
        predicted_labels = predicted_labels.detach().cpu().numpy()
    
    # Get unique classes
    unique_classes = np.unique(support_labels)
    n_way = len(unique_classes)
    
    # Set up the figure
    fig = plt.figure(figsize=figsize)
    
    # Calculate grid dimensions based on whether query set is provided
    if query_images is not None:
        outer_grid = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5], wspace=0.2)
        support_grid = gridspec.GridSpecFromSubplotSpec(n_way, max_examples, subplot_spec=outer_grid[0])
        query_grid = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[1])
        
        # Initialize query subplot
        query_ax = plt.Subplot(fig, query_grid[0])
        fig.add_subplot(query_ax)
        query_ax.set_title('Query Set (with Predictions)')
        query_ax.axis('off')
        
        # Create a grid for query images
        n_query = min(len(query_images), 20)  # Limit to 20 query images
        query_rows = min(4, n_query)
        query_cols = int(np.ceil(n_query / query_rows))
        inner_query_grid = gridspec.GridSpecFromSubplotSpec(
            query_rows, query_cols, subplot_spec=query_grid[0], wspace=0.1, hspace=0.1
        )
    else:
        # Just the support grid
        support_grid = gridspec.GridSpec(n_way, max_examples, wspace=0.1, hspace=0.1)
    
    # Function to preprocess image for display
    def preprocess_image(img):
        # Handle different image formats
        if img.ndim == 3:
            # Check if it's a grayscale image with a channel dimension (1, H, W)
            if img.shape[0] == 1:
                img = img.squeeze(0)  # Convert (1, H, W) to (H, W)
            # Or a color image in CHW format
            elif img.shape[0] == 3 and transpose_channels:
                img = np.transpose(img, (1, 2, 0))  # Convert (3, H, W) to (H, W, 3)
        
        # Apply normalization if specified
        if normalization:
            mean, std = normalization
            img = img * std + mean
        
        # Ensure values are in displayable range
        img = np.clip(img, 0, 1)
        
        return img
    
    # Plot support set
    for i, class_idx in enumerate(unique_classes):
        # Get images for this class
        mask = support_labels == class_idx
        class_images = support_images[mask]
        
        # Plot up to max_examples
        for j in range(min(len(class_images), max_examples)):
            ax = plt.Subplot(fig, support_grid[i, j])
            fig.add_subplot(ax)
            
            img = preprocess_image(class_images[j])
            
            if img.shape[-1] == 1:  # Grayscale
                ax.imshow(np.squeeze(img), cmap='gray')
            else:
                ax.imshow(img)
                
            ax.axis('off')
            
            if j == 0:
                ax.set_title(f'Class {class_idx}', fontsize=10)
    
    # Plot query set if provided
    if query_images is not None and query_labels is not None:
        query_count = 0
        for row in range(query_rows):
            for col in range(query_cols):
                if query_count < n_query:
                    ax = plt.Subplot(fig, inner_query_grid[row, col])
                    fig.add_subplot(ax)
                    
                    img = preprocess_image(query_images[query_count])
                    
                    if img.shape[-1] == 1:  # Grayscale
                        ax.imshow(np.squeeze(img), cmap='gray')
                    else:
                        ax.imshow(img)
                        
                    # Add border for correct/incorrect predictions
                    if predicted_labels is not None:
                        true_label = query_labels[query_count]
                        pred_label = predicted_labels[query_count]
                        
                        # Green for correct, red for incorrect
                        border_color = 'green' if true_label == pred_label else 'red'
                        
                        for spine in ax.spines.values():
                            spine.set_edgecolor(border_color)
                            spine.set_linewidth(2)
                            spine.set_visible(True)
                        
                        ax.set_title(f'T:{true_label} P:{pred_label}', fontsize=8)
                    
                    ax.axis('off')
                    query_count += 1
    
    plt.suptitle('Few-Shot Learning Task Examples', fontsize=16)
    return fig 