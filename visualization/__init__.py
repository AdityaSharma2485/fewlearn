"""
Visualization utilities for few-shot learning.

This package contains utilities for visualizing the results
of few-shot learning evaluations.
"""

from visualization.plotting import (
    plot_prototype_embeddings,
    plot_confusion_matrix,
    plot_performance_comparison,
    plot_task_examples
)

# Alias for plot_task_examples to make it easier to use in the app
def plot_support_query_sets(images, labels, **kwargs):
    """
    A wrapper around plot_task_examples that properly handles various image formats.
    
    This function ensures images are in the correct format for visualization,
    particularly handling grayscale images properly.
    
    Args:
        images: Support or query images
        labels: Corresponding labels
        **kwargs: Additional arguments to pass to plot_task_examples
        
    Returns:
        Matplotlib figure
    """
    import numpy as np
    import torch
    
    # Convert torch tensor to numpy if needed
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    
    # Check for grayscale images with channel dimension (1, H, W)
    # and reshape them to (H, W) to avoid matplotlib errors
    if images.ndim == 4 and images.shape[1] == 1:
        # Shape is (N, 1, H, W), reshape to (N, H, W)
        images = images.squeeze(1)
    
    return plot_task_examples(images, labels, **kwargs)

__all__ = [
    "plot_prototype_embeddings",
    "plot_confusion_matrix",
    "plot_performance_comparison",
    "plot_task_examples",
    "plot_support_query_sets"
] 
