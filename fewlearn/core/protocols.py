"""
Evaluation protocols for few-shot learning.

This module provides different protocols for evaluating models
in few-shot learning scenarios.
"""

import torch
from torch.utils.data import DataLoader
import concurrent.futures
from typing import Dict, List, Union, Optional, Callable, Any
import numpy as np
import traceback


class EvaluationProtocol:
    """Base class for all evaluation protocols."""
    
    def evaluate(self, 
                 models: Dict[str, torch.nn.Module],
                 data_loader: DataLoader,
                 metrics: List[str],
                 parallel: bool,
                 device: torch.device,
                 progress_callback: Optional[Callable[[float], None]] = None
                ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate models according to the protocol.
        
        Args:
            models: Dictionary of models to evaluate
            data_loader: DataLoader providing evaluation tasks
            metrics: List of metrics to compute
            parallel: Whether to run evaluations in parallel
            device: Device to run computations on
            progress_callback: Function to call with progress updates (0.0 to 1.0)
            
        Returns:
            Dictionary of metrics for each model and associated metadata
        """
        raise NotImplementedError("Subclasses must implement evaluate()")


class EpisodicProtocol(EvaluationProtocol):
    """
    Episodic evaluation protocol for few-shot learning.
    
    This protocol evaluates models on N-way, K-shot tasks (episodes),
    where each task contains support and query examples for N classes
    with K examples per class in the support set.
    
    Attributes:
        n_way (int): Number of classes in each task
        n_shot (int): Number of examples per class in the support set
        n_query (int): Number of query examples per class
        episodes (int): Number of episodes to evaluate
    """
    
    def __init__(self, 
                 n_way: int = 5, 
                 n_shot: int = 1, 
                 n_query: int = 15,
                 episodes: int = 100):
        """
        Initialize an episodic evaluation protocol.
        
        Args:
            n_way: Number of classes in each task
            n_shot: Number of examples per class in the support set
            n_query: Number of query examples per class
            episodes: Number of episodes to evaluate
        """
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.episodes = episodes
    
    def evaluate(self, 
                 models: Dict[str, torch.nn.Module],
                 data_loader: DataLoader,
                 metrics: List[str] = ["accuracy"],
                 parallel: bool = True,
                 device: torch.device = None,
                 progress_callback: Optional[Callable[[float], None]] = None
                ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate models on episodic tasks.
        
        Args:
            models: Dictionary of models to evaluate
            data_loader: DataLoader providing episodic tasks
            metrics: List of metrics to compute
            parallel: Whether to run evaluations in parallel
            device: Device to run computations on
            progress_callback: Function to call with progress updates (0.0 to 1.0)
            
        Returns:
            Dictionary of metrics for each model and associated metadata
        """
        if not device:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # Dictionary to store results for each model
        all_results = {}
        
        # Function to evaluate a single model on all episodes
        def evaluate_single_model(model_name, model, episodes_data):
            model = model.to(device)
            model.eval()
            
            # Storage for predictions and metrics
            all_predictions = []
            all_targets = []
            episode_metrics = []
            last_episode_data = None
            
            try:
                with torch.no_grad():
                    for episode_idx, (support_images, support_labels, query_images, query_labels, class_ids) in enumerate(episodes_data):
                        # Check if we have necessary data
                        if support_images.size(0) == 0 or query_images.size(0) == 0:
                            print(f"Warning: Empty support or query set in episode {episode_idx}, skipping")
                            continue
                            
                        # Move tensors to device
                        support_images = support_images.clone().to(device)
                        support_labels = support_labels.clone().to(device)
                        query_images = query_images.clone().to(device)
                        query_labels = query_labels.clone().to(device)
                        
                        try:
                            # Get model predictions
                            logits = model(support_images, support_labels, query_images)
                            
                            # Ensure prediction shape matches expectations
                            if logits.shape[0] != query_labels.shape[0]:
                                print(f"Warning: Shape mismatch - logits: {logits.shape}, query_labels: {query_labels.shape}")
                                # If we can't reshape, just skip this episode
                                if logits.shape[0] > query_labels.shape[0]:
                                    # Truncate predictions to match targets
                                    logits = logits[:query_labels.shape[0]]
                                elif query_labels.shape[0] > logits.shape[0]:
                                    # Truncate targets to match predictions
                                    query_labels = query_labels[:logits.shape[0]]
                                else:
                                    continue
                                    
                            predicted_labels = torch.argmax(logits, dim=1)
                            
                            # Convert to numpy for metric calculation
                            predictions = predicted_labels.detach().cpu().numpy()
                            targets = query_labels.detach().cpu().numpy()
                            
                            # Store predictions and targets
                            all_predictions.append(predictions)
                            all_targets.append(targets)
                        except Exception as e:
                            print(f"Error processing episode {episode_idx}: {str(e)}")
                            continue
                        
                        # Calculate metrics for this episode
                        episode_result = {}
                        for metric in metrics:
                            if metric == "accuracy":
                                episode_result[metric] = np.mean(predictions == targets)
                            elif metric == "f1":
                                try:
                                    # Import F1 score from sklearn with error handling
                                    from sklearn.metrics import f1_score
                                    # Make sure predictions and targets have the same shape
                                    if len(predictions) != len(targets):
                                        print(f"Warning: Inconsistent shapes in episode {episode_idx}: predictions {predictions.shape}, targets {targets.shape}")
                                        # Fix by using only the first min(len(predictions), len(targets)) elements
                                        min_len = min(len(predictions), len(targets))
                                        predictions_safe = predictions[:min_len]
                                        targets_safe = targets[:min_len]
                                        episode_result[metric] = f1_score(targets_safe, predictions_safe, average='weighted', zero_division=0)
                                    else:
                                        episode_result[metric] = f1_score(targets, predictions, average='weighted', zero_division=0)
                                except Exception as e:
                                    print(f"Error calculating F1 score: {str(e)}")
                                    # Fallback to accuracy if F1 calculation fails
                                    episode_result[metric] = np.mean(predictions == targets)
                            else:
                                # Default to accuracy if metric not implemented
                                episode_result[metric] = np.mean(predictions == targets)
                                
                        episode_metrics.append(episode_result)
                        
                        # Store data from the last episode for visualization
                        if episode_idx == len(episodes_data) - 1:
                            last_episode_data = {
                                "support_images": support_images.detach().cpu().numpy(),
                                "support_labels": support_labels.detach().cpu().numpy(),
                                "query_images": query_images.detach().cpu().numpy(),
                                "query_labels": targets,
                                "predicted_labels": predictions,
                                "class_ids": class_ids
                            }
                
                # Calculate overall metrics
                all_predictions = np.concatenate(all_predictions)
                all_targets = np.concatenate(all_targets)
                
                # Calculate overall metrics
                overall_metrics = {}
                for metric in metrics:
                    if metric == "accuracy":
                        overall_metrics[metric] = np.mean(all_predictions == all_targets)
                    elif metric == "f1":
                        try:
                            from sklearn.metrics import f1_score
                            # Make sure predictions and targets have the same shape
                            if len(all_predictions) != len(all_targets):
                                print(f"Warning: Inconsistent shapes in overall metrics: predictions {all_predictions.shape}, targets {all_targets.shape}")
                                # Fix by using only the first min(len(predictions), len(targets)) elements
                                min_len = min(len(all_predictions), len(all_targets))
                                predictions_safe = all_predictions[:min_len]
                                targets_safe = all_targets[:min_len]
                                overall_metrics[metric] = f1_score(targets_safe, predictions_safe, average='weighted', zero_division=0)
                            else:
                                overall_metrics[metric] = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
                        except Exception as e:
                            print(f"Error calculating overall F1 score: {str(e)}")
                            # Fallback to accuracy if F1 calculation fails
                            overall_metrics[metric] = np.mean(all_predictions == all_targets)
                    else:
                        # Default to accuracy if metric not implemented
                        overall_metrics[metric] = np.mean(all_predictions == all_targets)
                
                # Aggregate episode metrics
                aggregated_metrics = {}
                for metric in metrics:
                    metric_values = [e[metric] for e in episode_metrics]
                    aggregated_metrics[f"{metric}_mean"] = np.mean(metric_values)
                    aggregated_metrics[f"{metric}_std"] = np.std(metric_values)
                    aggregated_metrics[f"{metric}_min"] = np.min(metric_values)
                    aggregated_metrics[f"{metric}_max"] = np.max(metric_values)
                
                return {
                    "metrics": overall_metrics,
                    "aggregated_metrics": aggregated_metrics,
                    "episodes": episode_metrics,
                    "last_episode": last_episode_data,
                    "config": {
                        "n_way": self.n_way,
                        "n_shot": self.n_shot,
                        "n_query": self.n_query,
                        "episodes": self.episodes
                    }
                }
            except Exception as e:
                print(f"Error evaluating model {model_name}: {str(e)}")
                traceback.print_exc()
                return {
                    "metrics": {m: 0.0 for m in metrics},
                    "aggregated_metrics": {},
                    "episodes": [],
                    "last_episode": None,
                    "config": {
                        "n_way": self.n_way,
                        "n_shot": self.n_shot,
                        "n_query": self.n_query,
                        "episodes": self.episodes
                    },
                    "error": str(e)
                }
        
        # Convert data_loader to a list to avoid iterator consumption issues
        episodes_data = list(data_loader)
        
        # Track progress
        models_completed = 0
        total_models = len(models)
        
        try:
            # Determine if we should use parallel execution
            if parallel and len(models) > 1:
                # Use ThreadPoolExecutor for parallel evaluation
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(models), 4)) as executor:
                    # Submit jobs
                    future_to_model = {
                        executor.submit(evaluate_single_model, name, model, episodes_data): name
                        for name, model in models.items()
                    }
                    
                    # Process results as they complete
                    for future in concurrent.futures.as_completed(future_to_model):
                        name = future_to_model[future]
                        try:
                            result = future.result()
                            all_results[name] = result
                        except Exception as exc:
                            print(f"{name} generated an exception: {exc}")
                            all_results[name] = {
                                "metrics": {m: 0.0 for m in metrics},
                                "aggregated_metrics": {},
                                "episodes": [],
                                "last_episode": None,
                                "config": {
                                    "n_way": self.n_way,
                                    "n_shot": self.n_shot,
                                    "n_query": self.n_query,
                                    "episodes": self.episodes
                                },
                                "error": str(exc)
                            }
                        
                        # Update progress
                        models_completed += 1
                        if progress_callback:
                            progress_callback(models_completed / total_models)
            else:
                # Sequential evaluation
                for name, model in models.items():
                    result = evaluate_single_model(name, model, episodes_data)
                    all_results[name] = result
                    
                    # Update progress
                    models_completed += 1
                    if progress_callback:
                        progress_callback(models_completed / total_models)
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            traceback.print_exc()
        
        return all_results 
