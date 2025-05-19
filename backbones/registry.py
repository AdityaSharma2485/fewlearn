"""
Registry for backbone models.

This module provides a central registry for backbone models that can be
used in few-shot learning, allowing dynamic registration and retrieval.
"""

import torch
from torch import nn
from typing import Dict, Callable, Optional, Union

# Global registry of backbone models
_BACKBONE_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_backbone(name: str, constructor: Callable[..., nn.Module]) -> None:
    """
    Register a backbone model constructor.
    
    Args:
        name: Unique identifier for the backbone
        constructor: Function that constructs the backbone model
    """
    if name in _BACKBONE_REGISTRY:
        raise ValueError(f"Backbone '{name}' is already registered")
    
    _BACKBONE_REGISTRY[name] = constructor


def get_backbone(name: str, **kwargs) -> nn.Module:
    """
    Get a backbone model by name.
    
    Args:
        name: Name of the backbone model
        **kwargs: Additional arguments to pass to the constructor
        
    Returns:
        Instantiated backbone model
    """
    if name not in _BACKBONE_REGISTRY:
        raise ValueError(f"Backbone '{name}' is not registered")
    
    return _BACKBONE_REGISTRY[name](**kwargs)


def list_backbones() -> list:
    """
    List all registered backbone models.
    
    Returns:
        List of backbone model names
    """
    return list(_BACKBONE_REGISTRY.keys())


# Register common torchvision models as backbones
def _register_torchvision_models():
    """Register common models from torchvision as backbones."""
    try:
        import torchvision.models as models
        
        # ResNet family
        def resnet18(pretrained=True):
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet18(weights=weights)
            # Replace the final classification layer with an identity
            model.fc = nn.Flatten()
            return model
        
        def resnet50(pretrained=True):
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet50(weights=weights)
            model.fc = nn.Flatten()
            return model
        
        # MobileNet
        def mobilenet_v2(pretrained=True):
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.mobilenet_v2(weights=weights)
            model.classifier = nn.Flatten()
            return model
        
        # EfficientNet
        def efficientnet_b0(pretrained=True):
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            model.classifier = nn.Flatten()
            return model
        
        # DenseNet
        def densenet121(pretrained=True):
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.densenet121(weights=weights)
            model.classifier = nn.Flatten()
            return model
        
        # VGG
        def vgg16(pretrained=True):
            weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.vgg16(weights=weights)
            model.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout()
            )
            return model
            
        # GoogleNet / Inception
        def googlenet(pretrained=True):
            weights = models.GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.googlenet(weights=weights)
            model.fc = nn.Flatten()
            return model
        
        # Register all models
        register_backbone("resnet18", resnet18)
        register_backbone("resnet50", resnet50)
        register_backbone("mobilenet_v2", mobilenet_v2)
        register_backbone("efficientnet_b0", efficientnet_b0)
        register_backbone("densenet121", densenet121)
        register_backbone("vgg16", vgg16)
        register_backbone("googlenet", googlenet)
        
    except ImportError:
        print("Warning: torchvision is not available. Standard backbones will not be registered.")


# Register custom models
def _register_custom_models():
    """Register custom backbone models."""
    # Import and register custom backbone models here
    pass


# Initialize the registry
_register_torchvision_models()
_register_custom_models() 
