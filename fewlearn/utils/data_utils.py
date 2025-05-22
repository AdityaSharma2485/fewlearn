"""
Data utilities for few-shot learning.

This module provides utilities for working with datasets in few-shot
learning, including creating episodic data loaders.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Union, Optional, Dict, Any, List, Tuple
import numpy as np

try:
    from easyfsl.samplers import TaskSampler
    HAS_EASYFSL = True
except ImportError:
    HAS_EASYFSL = False


def create_episode_loader(
    dataset: Dataset,
    n_way: int = 5,
    n_shot: int = 1,
    n_query: int = 15,
    n_episodes: int = 100,
    batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a data loader for episodic few-shot learning.
    
    Args:
        dataset: The dataset to sample episodes from
        n_way: Number of classes in each episode
        n_shot: Number of support examples per class
        n_query: Number of query examples per class
        n_episodes: Number of episodes
        batch_size: Batch size (if None, n_way * (n_shot + n_query))
        num_workers: Number of workers for loading data
        pin_memory: Whether to pin memory for GPU transfer
        
    Returns:
        DataLoader for episodic evaluation
    """
    # First, ensure the dataset has a get_labels method
    if not hasattr(dataset, 'get_labels') and not callable(getattr(dataset, 'get_labels', None)):
        # Try to infer the correct way to get labels
        if hasattr(dataset, 'targets'):
            # Many PyTorch datasets have a targets attribute
            dataset.get_labels = lambda: dataset.targets
        elif hasattr(dataset, 'labels'):
            # Some datasets have a labels attribute
            dataset.get_labels = lambda: dataset.labels
        elif hasattr(dataset, '_flat_character_images'):
            # Omniglot specific
            dataset.get_labels = lambda: [instance[1] for instance in dataset._flat_character_images]
        else:
            # As a last resort, try to infer from __getitem__
            try:
                sample = dataset[0]
                if isinstance(sample, tuple) and len(sample) >= 2:
                    # Assuming the second element is the label
                    dataset.get_labels = lambda: [dataset[i][1] for i in range(len(dataset))]
                else:
                    raise ValueError("Could not infer labels from dataset")
            except:
                raise ValueError("Dataset must have a get_labels method or a recognizable label attribute")
    
    # Get the labels
    try:
        labels = dataset.get_labels()
        # Ensure labels are in a usable format
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        elif not isinstance(labels, (list, np.ndarray)):
            # Convert to list if it's some other type
            labels = list(labels)
    except Exception as e:
        print(f"Error getting labels: {str(e)}")
        raise ValueError(f"Failed to get dataset labels: {str(e)}")
    
    # Create our sampler
    try:
        from fewlearn.utils.samplers import EpisodicSampler
        
        if batch_size is None:
            batch_size = n_way * (n_shot + n_query)
            
        sampler = EpisodicSampler(
            labels, 
            n_way=n_way, 
            n_shot=n_shot, 
            n_query=n_query, 
            n_episodes=n_episodes
        )
        
        # Create data loader with our custom sampler and collate function
        data_loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=sampler.episodic_collate_fn
        )
        
        return data_loader
    except Exception as e:
        print(f"Error creating episode loader: {str(e)}")
        raise ValueError(f"Failed to create episode loader: {str(e)}")


def prepare_batch_for_model(batch: Tuple, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare a batch of data for a few-shot learning model.
    
    Args:
        batch: Tuple from the data loader (support_images, support_labels, query_images, query_labels, ...)
        device: Device to move tensors to
        
    Returns:
        Tuple of (support_images, support_labels, query_images) on the device
    """
    support_images, support_labels, query_images = batch[0], batch[1], batch[2]
    
    support_images = support_images.to(device)
    support_labels = support_labels.to(device)
    query_images = query_images.to(device)
    
    return support_images, support_labels, query_images 
