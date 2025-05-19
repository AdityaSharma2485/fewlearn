"""
Evaluator for few-shot learning models.

This module provides the main interface for evaluating few-shot
learning models using various protocols and metrics.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Union, Optional, Callable, Any
import numpy as np
import pandas as pd
import time
import json
from pathlib import Path

# Import only EpisodicProtocol to avoid circular imports
from fewlearn.core.protocols import EpisodicProtocol
from fewlearn.models.base import FewShotModel


class Evaluator:
    """
    Main evaluator class for few-shot learning models.
    
    The evaluator provides a high-level interface for evaluating and
    comparing multiple models using various protocols and metrics.
    
    Attributes:
        protocol: The evaluation protocol to use
        metrics: List of metrics to compute
        parallel: Whether to run evaluations in parallel
    """
    
    def __init__(self, 
                 protocol: Optional[Any] = None,
                 metrics: List[str] = ["accuracy"],
                 parallel: bool = True):
        """
        Initialize an evaluator.
        
        Args:
            protocol: The evaluation protocol to use
            metrics: List of metrics to compute
            parallel: Whether to run evaluations in parallel
        """
        self.protocol = protocol or EpisodicProtocol()
        self.metrics = metrics
        self.parallel = parallel
        self.results = None
    
    def evaluate(self, 
                 models: Dict[str, FewShotModel],
                 dataset: torch.utils.data.Dataset,
                 seed: Optional[int] = None,
                 progress_callback: Optional[Callable[[float], None]] = None
                ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate one or more models on a dataset.
        
        Args:
            models: Dictionary of models to evaluate
            dataset: Dataset to evaluate on
            seed: Random seed for reproducibility
            progress_callback: Function to call with progress updates
            
        Returns:
            Dictionary of evaluation results
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Create data loader from protocol
        if hasattr(self.protocol, 'create_data_loader'):
            data_loader = self.protocol.create_data_loader(dataset)
        else:
            # Use default data loader if protocol doesn't provide one
            from fewlearn.utils.data_utils import create_episode_loader
            data_loader = create_episode_loader(
                dataset, 
                n_way=getattr(self.protocol, 'n_way', 5),
                n_shot=getattr(self.protocol, 'n_shot', 1),
                n_query=getattr(self.protocol, 'n_query', 15),
                n_episodes=getattr(self.protocol, 'episodes', 100)
            )
            
        # Record start time
        start_time = time.time()
        
        # Evaluate models using the protocol
        results = self.protocol.evaluate(
            models=models,
            data_loader=data_loader,
            metrics=self.metrics,
            parallel=self.parallel,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            progress_callback=progress_callback
        )
        
        # Record end time
        end_time = time.time()
        
        # Add evaluation metadata
        for model_name in results:
            results[model_name]["metadata"] = {
                "model_name": model_name,
                "evaluation_time": end_time - start_time,
                "timestamp": time.time(),
                "metrics": self.metrics,
                "parallel": self.parallel
            }
        
        # Store results
        self.results = results
        
        return results
    
    def summary(self) -> pd.DataFrame:
        """
        Create a summary DataFrame of the evaluation results.
        
        Returns:
            DataFrame with model names as index and metrics as columns
        """
        if self.results is None:
            raise ValueError("No evaluation results available. Run evaluate() first.")
            
        # Extract metrics from results
        data = []
        for model_name, result in self.results.items():
            row = {"model": model_name}
            
            # Add main metrics
            for metric, value in result["metrics"].items():
                row[metric] = value
                
            # Add aggregated metrics if available
            if "aggregated_metrics" in result:
                for agg_metric, value in result["aggregated_metrics"].items():
                    row[agg_metric] = value
                    
            # Add evaluation time if available
            if "metadata" in result and "evaluation_time" in result["metadata"]:
                row["evaluation_time"] = result["metadata"]["evaluation_time"]
                
            data.append(row)
            
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Set model as index if it exists
        if "model" in df.columns:
            df = df.set_index("model")
            
        return df
    
    def save_results(self, filename: str) -> None:
        """
        Save evaluation results to a file.
        
        Args:
            filename: Path to save the results to
        """
        if self.results is None:
            raise ValueError("No evaluation results available. Run evaluate() first.")
            
        # Convert numpy arrays to lists for JSON serialization
        def convert_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(item) for item in obj]
            else:
                return obj
                
        serializable_results = convert_arrays(self.results)
        
        # Create directory if it doesn't exist
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on file extension
        if filename.endswith('.json'):
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        elif filename.endswith('.csv'):
            self.summary().to_csv(filename)
        else:
            # Default to JSON
            with open(f"{filename}.json", 'w') as f:
                json.dump(serializable_results, f, indent=2)
    
    def load_results(self, filename: str) -> Dict[str, Dict[str, Any]]:
        """
        Load evaluation results from a file.
        
        Args:
            filename: Path to load the results from
            
        Returns:
            Loaded results
        """
        if filename.endswith('.json'):
            with open(filename, 'r') as f:
                self.results = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
            
        return self.results 