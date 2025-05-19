"""
Samplers for few-shot learning data loaders.

This module provides samplers for few-shot learning, including
episodic task samplers for N-way, K-shot tasks.
"""

import torch
from torch.utils.data import Sampler
import numpy as np
from typing import List, Iterator, Sequence, Dict, Optional


class EpisodicSampler(Sampler):
    """
    Sampler for episodic few-shot learning.
    
    This sampler yields batches for N-way, K-shot classification tasks,
    where each task contains support and query examples for N classes
    with K examples per class in the support set.
    
    Attributes:
        labels: Labels for the entire dataset
        n_way: Number of classes in each task
        n_shot: Number of examples per class in the support set
        n_query: Number of query examples per class
        n_episodes: Number of episodes (tasks) to generate
    """
    
    def __init__(self, 
                 labels: Sequence[int],
                 n_way: int = 5,
                 n_shot: int = 1,
                 n_query: int = 15,
                 n_episodes: int = 100):
        """
        Initialize an episodic sampler.
        
        Args:
            labels: Labels for the entire dataset
            n_way: Number of classes in each task
            n_shot: Number of examples per class in the support set
            n_query: Number of query examples per class
            n_episodes: Number of episodes to generate
        """
        self.labels = np.array(labels)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        
        # Group indices by label
        self.label_indices = {}
        unique_labels = np.unique(self.labels)
        
        for label in unique_labels:
            self.label_indices[label] = np.where(self.labels == label)[0]
            
        # Ensure we have enough classes
        if len(unique_labels) < self.n_way:
            raise ValueError(f"Dataset has only {len(unique_labels)} classes, "
                            f"but {self.n_way} are required for n_way")
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Generate episode batches.
        
        Each batch contains indices for support and query examples
        for a single episode.
        
        Returns:
            Iterator over batches of indices
        """
        for _ in range(self.n_episodes):
            # Randomly select n_way classes
            episode_classes = np.random.choice(
                list(self.label_indices.keys()),
                self.n_way,
                replace=False
            )
            
            # Create a batch of indices for this episode
            batch_indices = []
            
            # Process one class at a time to maintain proper order
            for class_idx, class_label in enumerate(episode_classes):
                # Get indices for this class
                class_indices = self.label_indices[class_label]
                
                # Ensure we have enough examples
                if len(class_indices) < (self.n_shot + self.n_query):
                    # If not enough examples, sample with replacement
                    sampled_indices = np.random.choice(
                        class_indices,
                        self.n_shot + self.n_query,
                        replace=True
                    )
                else:
                    # Sample without replacement
                    sampled_indices = np.random.choice(
                        class_indices,
                        self.n_shot + self.n_query,
                        replace=False
                    )
                
                # Support indices go first (for all classes)
                support_indices = sampled_indices[:self.n_shot]
                batch_indices.extend(support_indices)
            
            # Then add query indices (for all classes)
            for class_idx, class_label in enumerate(episode_classes):
                # Get indices for this class
                class_indices = self.label_indices[class_label]
                
                # Ensure we use the same sampling logic for consistency
                if len(class_indices) < (self.n_shot + self.n_query):
                    sampled_indices = np.random.choice(
                        class_indices,
                        self.n_shot + self.n_query,
                        replace=True
                    )
                else:
                    sampled_indices = np.random.choice(
                        class_indices,
                        self.n_shot + self.n_query,
                        replace=False
                    )
                
                # Add query indices after all support indices
                query_indices = sampled_indices[self.n_shot:(self.n_shot + self.n_query)]
                batch_indices.extend(query_indices)
            
            # Convert to integer list and yield
            yield [int(idx) for idx in batch_indices]
    
    def __len__(self) -> int:
        """
        Get the number of episodes.
        
        Returns:
            Number of episodes
        """
        return self.n_episodes
    
    def episodic_collate_fn(self, batch):
        """
        Custom collate function for episodic batches.
        
        This formats the data into support and query sets.
        
        Args:
            batch: Batch of data from DataLoader
            
        Returns:
            Tuple of (support_x, support_y, query_x, query_y, class_ids)
        """
        # Get the dimensions
        n_way, n_shot, n_query = self.n_way, self.n_shot, self.n_query
        support_size = n_way * n_shot
        query_size = n_way * n_query
        
        # Ensure batch is the expected size
        if len(batch) != (support_size + query_size):
            print(f"Warning: Batch size {len(batch)} doesn't match expected {support_size + query_size}")
            # Adjust sizes to match what we have
            if len(batch) < support_size:
                # Not enough even for support set
                actual_support_size = len(batch)
                actual_query_size = 0
            else:
                # We have enough for support, but maybe not full query set
                actual_support_size = support_size
                actual_query_size = min(len(batch) - support_size, query_size)
        else:
            actual_support_size = support_size
            actual_query_size = query_size
            
        # Extract the data and labels
        data = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        # Convert to tensors
        if not data:
            # Empty batch case
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), []
        
        data = torch.stack(data)
        labels = torch.tensor(labels)
        
        # Split into support and query
        support_x = data[:actual_support_size]
        support_y = labels[:actual_support_size]
        
        query_x = data[actual_support_size:actual_support_size + actual_query_size] if actual_query_size > 0 else torch.tensor([])
        query_y = labels[actual_support_size:actual_support_size + actual_query_size] if actual_query_size > 0 else torch.tensor([])
        
        # Extract unique class IDs from support set
        class_ids = torch.unique(support_y).tolist()
        
        # Create a mapping from original class IDs to [0, n_way)
        # This is crucial for model's expected labels
        if class_ids:
            label_map = {class_ids[i]: i for i in range(len(class_ids))}
            
            # Remap support labels
            remapped_support_y = torch.tensor([label_map.get(y.item(), 0) for y in support_y])
            
            # Remap query labels if we have any
            if len(query_y) > 0:
                remapped_query_y = torch.tensor([
                    label_map.get(y.item(), 0) if y.item() in label_map else 0 
                    for y in query_y
                ])
            else:
                remapped_query_y = query_y
                
            return support_x, remapped_support_y, query_x, remapped_query_y, class_ids
        else:
            # Empty class_ids case - return original labels
            return support_x, support_y, query_x, query_y, [] 