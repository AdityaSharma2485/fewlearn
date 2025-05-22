"""
Feature extraction backbones for few-shot learning models.

This package contains implementations and adapters for various
backbone architectures used in few-shot learning.
"""

from fewlearn.backbones.registry import register_backbone, get_backbone

__all__ = ["register_backbone", "get_backbone"] 