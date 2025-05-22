"""
Metrics for evaluating few-shot learning models.

This module provides a collection of metrics for evaluating the performance
of few-shot learning models, including accuracy, precision, recall, and F1 score.
"""

import numpy as np
from typing import Dict, List, Union, Optional, Callable, Any
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score as sklearn_f1_score,
    roc_auc_score
)


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate accuracy.
    
    Args:
        predictions: Predicted labels
        targets: Ground truth labels
        
    Returns:
        Accuracy score
    """
    return accuracy_score(targets, predictions)


def precision(predictions: np.ndarray, targets: np.ndarray, average: str = 'weighted') -> float:
    """
    Calculate precision.
    
    Args:
        predictions: Predicted labels
        targets: Ground truth labels
        average: Averaging strategy
        
    Returns:
        Precision score
    """
    return precision_score(targets, predictions, average=average, zero_division=0)


def recall(predictions: np.ndarray, targets: np.ndarray, average: str = 'weighted') -> float:
    """
    Calculate recall.
    
    Args:
        predictions: Predicted labels
        targets: Ground truth labels
        average: Averaging strategy
        
    Returns:
        Recall score
    """
    return recall_score(targets, predictions, average=average, zero_division=0)


def f1_score(predictions: np.ndarray, targets: np.ndarray, average: str = 'weighted') -> float:
    """
    Calculate F1 score.
    
    Args:
        predictions: Predicted labels
        targets: Ground truth labels
        average: Averaging strategy
        
    Returns:
        F1 score
    """
    return sklearn_f1_score(targets, predictions, average=average, zero_division=0)


def auc_roc(probabilities: np.ndarray, targets: np.ndarray, multi_class: str = 'ovr') -> float:
    """
    Calculate Area Under the ROC Curve.
    
    Args:
        probabilities: Predicted probabilities for each class
        targets: Ground truth labels
        multi_class: Strategy for multi-class classification
        
    Returns:
        AUC-ROC score
    """
    try:
        # Check if there are at least two unique classes
        if len(np.unique(targets)) < 2:
            return float('nan')
            
        return roc_auc_score(targets, probabilities, multi_class=multi_class)
    except ValueError:
        # Handle the case where AUC-ROC is not defined
        return float('nan')


# Registry of available metrics
_METRICS = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1_score,
    'auc_roc': auc_roc
}


def register_metric(name: str, metric_fn: Callable) -> None:
    """
    Register a custom metric.
    
    Args:
        name: Name of the metric
        metric_fn: Function that calculates the metric
    """
    if name in _METRICS:
        raise ValueError(f"Metric '{name}' is already registered")
        
    _METRICS[name] = metric_fn


def calculate_metrics(predictions: np.ndarray, 
                     targets: np.ndarray, 
                     metrics: List[str] = None,
                     probabilities: Optional[np.ndarray] = None
                    ) -> Dict[str, float]:
    """
    Calculate multiple metrics.
    
    Args:
        predictions: Predicted labels or indices
        targets: Ground truth labels or indices
        metrics: List of metrics to calculate (None for all metrics except auc_roc)
        probabilities: Predicted probabilities (for metrics like AUC-ROC)
        
    Returns:
        Dictionary of metric names and values
    """
    results = {}
    
    # Default to all metrics except auc_roc which requires probabilities
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Make sure predictions and targets are 1D arrays
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        # Predictions are in one-hot format or scores, convert to indices
        predictions = np.argmax(predictions, axis=1)

    if targets.ndim > 1 and targets.shape[1] > 1:
        # Targets are in one-hot format, convert to indices
        targets = np.argmax(targets, axis=1)

    # Ensure predictions and targets are in the same shape
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Convert predictions and targets to numpy arrays if they're not already
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    # Ensure both are arrays of integers for classification metrics
    predictions = predictions.astype(np.int64)
    targets = targets.astype(np.int64)
    
    # Ensure predictions and targets have matching shapes
    if len(predictions) != len(targets):
        raise ValueError(f"Predictions and targets must have the same length: {len(predictions)} vs {len(targets)}")
    
    for metric in metrics:
        if metric not in _METRICS:
            raise ValueError(f"Unknown metric: {metric}")
            
        if metric == 'auc_roc':
            if probabilities is None:
                results[metric] = float('nan')
            else:
                results[metric] = _METRICS[metric](probabilities, targets)
        else:
            try:
                # Try with predictions first (the intended order)
                results[metric] = _METRICS[metric](predictions, targets)
            except Exception as e:
                try:
                    # If that fails, print more info about the issue
                    print(f"Error calculating {metric}: {str(e)}")
                    print(f"Predictions shape: {predictions.shape}, values: {predictions[:5]} (first 5)")
                    print(f"Targets shape: {targets.shape}, values: {targets[:5]} (first 5)")
                    # Let the error propagate up
                    raise
                except:
                    raise
            
    # Add confusion matrix
    if 'accuracy' in results and len(np.unique(targets)) > 1:
        try:
            from sklearn.metrics import confusion_matrix
            # Get unique classes including both predictions and targets
            all_classes = np.unique(np.concatenate([targets, predictions]))
            # Create the confusion matrix
            cm = confusion_matrix(targets, predictions, labels=all_classes)
            results['confusion_matrix'] = cm
        except Exception as e:
            print(f"Warning: Could not compute confusion matrix: {str(e)}")
    
    return results


def list_metrics() -> List[str]:
    """
    List all available metrics.
    
    Returns:
        List of metric names
    """
    return list(_METRICS.keys()) 
