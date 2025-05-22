"""
FewLearn: A Python module for few-shot learning with pretrained models

FewLearn provides tools for comparing and evaluating pretrained models
using few-shot learning techniques like Prototypical Networks.
"""

__version__ = "0.1.0"

# Import core components
from fewlearn.core.minds import MINDS
from fewlearn.core.protocols import EpisodicProtocol
from fewlearn.evaluation.evaluator import Evaluator

# Import models
from fewlearn.models.prototypical import PrototypicalNetworks

# Export key components
__all__ = [
    "MINDS",
    "PrototypicalNetworks",
    "Evaluator",
    "EpisodicProtocol",
] 