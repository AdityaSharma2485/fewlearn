"""
Base classes for few-shot learning models.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, Optional, Dict, Any

from backbones.registry import get_backbone


class FewShotModel(nn.Module):
    """
    Base class for all few-shot learning models.
    
    This class defines the interface that all few-shot learning models
    should implement to work with the MINDS framework.
    """
    
    def __init__(self, backbone: Union[str, nn.Module], **kwargs):
        """
        Initialize a few-shot learning model.
        
        Args:
            backbone: Either a string identifier for a registered backbone,
                     or a nn.Module to use as the backbone
            **kwargs: Additional arguments for the model
        """
        super().__init__()
        
        if isinstance(backbone, str):
            self.backbone = get_backbone(backbone)
        else:
            self.backbone = backbone
            
        # Store configuration
        self.config = {
            "backbone": backbone if isinstance(backbone, str) else backbone.__class__.__name__,
            **kwargs
        }
    
    def forward(self, 
                support_images: torch.Tensor,
                support_labels: torch.Tensor,
                query_images: torch.Tensor
               ) -> torch.Tensor:
        """
        Perform few-shot classification.
        
        Args:
            support_images: Tensor of shape [n_way * n_shot, channels, height, width]
                           containing the support images
            support_labels: Tensor of shape [n_way * n_shot] containing the support labels
            query_images: Tensor of shape [n_way * n_query, channels, height, width]
                         containing the query images
                         
        Returns:
            Tensor of shape [n_way * n_query, n_way] containing classification logits
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from inputs using the backbone.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Feature tensor
        """
        return self.backbone(x)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the model configuration.
        
        Returns:
            Dictionary containing model configuration
        """
        return self.config
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of the feature embeddings.
        
        Returns:
            Embedding dimension
        """
        # This is a simple heuristic that works for many backbones
        # Subclasses might need to override this
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224, device=next(self.parameters()).device)
            features = self.extract_features(dummy_input)
            return features.shape[1] 
