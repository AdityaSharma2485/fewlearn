"""
Prototypical Networks implementation for few-shot learning.

Reference:
    Snell, J., Swersky, K., & Zemel, R. (2017).
    Prototypical networks for few-shot learning.
    In Advances in neural information processing systems (pp. 4077-4087).
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, Optional, Dict, Any

from fewlearn.models.base import FewShotModel


class PrototypicalNetworks(FewShotModel):
    """
    Implementation of Prototypical Networks for few-shot learning.
    
    Prototypical Networks learn a metric space in which classification
    is performed by computing distances to prototype representations of
    each class.
    
    Attributes:
        backbone: The feature extractor
        distance: Distance function to use ('euclidean' or 'cosine')
    """
    
    def __init__(self, 
                 backbone: Union[str, nn.Module],
                 distance: str = 'euclidean',
                 **kwargs):
        """
        Initialize a Prototypical Networks model.
        
        Args:
            backbone: Feature extractor (string identifier or nn.Module)
            distance: Distance function to use ('euclidean' or 'cosine')
            **kwargs: Additional arguments passed to the backbone
        """
        super().__init__(backbone=backbone, distance=distance, **kwargs)
        self.distance = distance
    
    def forward(self,
                support_images: torch.Tensor,
                support_labels: torch.Tensor,
                query_images: torch.Tensor
               ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        
        Args:
            support_images: Tensor of shape [n_way * n_shot, channels, height, width]
                           containing the support images
            support_labels: Tensor of shape [n_way * n_shot] containing the support labels
            query_images: Tensor of shape [n_way * n_query, channels, height, width]
                         containing the query images
                         
        Returns:
            Tensor of shape [n_way * n_query, n_way] containing classification logits
        """
        # Extract features
        z_support = self.extract_features(support_images)
        z_query = self.extract_features(query_images)
        
        # Get unique labels and create a mapping to consecutive indices
        unique_labels = torch.unique(support_labels, sorted=True)
        n_way = len(unique_labels)
        
        # Create a mapping from original labels to consecutive indices (0 to n_way-1)
        label_to_idx = {label.item(): idx for idx, label in enumerate(unique_labels)}
        
        # Compute prototypes for each unique class
        z_proto = []
        for label in unique_labels:
            # Find support samples for this class
            class_mask = (support_labels == label)
            # Compute the prototype (mean of support features)
            class_proto = z_support[class_mask].mean(0, keepdim=True)
            z_proto.append(class_proto)
        
        # Stack prototypes
        z_proto = torch.cat(z_proto, dim=0)
        
        # Compute distances
        if self.distance == 'cosine':
            # Normalize features for cosine similarity
            z_query = F.normalize(z_query, dim=1)
            z_proto = F.normalize(z_proto, dim=1)
            
            # Compute cosine similarity (dot product of normalized vectors)
            scores = torch.mm(z_query, z_proto.t())
        else:  # euclidean
            # Compute euclidean distance
            dists = torch.cdist(z_query, z_proto)
            
            # Convert distances to scores (negative distance)
            scores = -dists
        
        return scores
    
    def compute_prototypes(self, 
                           support_images: torch.Tensor, 
                           support_labels: torch.Tensor
                          ) -> torch.Tensor:
        """
        Compute class prototypes from support images.
        
        Args:
            support_images: Support images tensor
            support_labels: Support labels tensor
            
        Returns:
            Class prototypes tensor
        """
        z_support = self.extract_features(support_images)
        
        # Get unique labels
        unique_labels = torch.unique(support_labels)
        
        # Compute prototype for each unique class
        z_proto = torch.cat(
            [
                z_support[support_labels == label].mean(0, keepdim=True)
                for label in unique_labels
            ]
        )
        
        return z_proto 