"""
FewLearn: A Python module for few-shot learning with pretrained models

FewLearn provides tools for comparing and evaluating pretrained models
using few-shot learning techniques like Prototypical Networks.
"""

__version__ = "0.1.1"

# Use lazy imports to avoid circular import issues
# This allows users to do "from fewlearn import MINDS" etc.
# while avoiding circular dependencies

__all__ = [
    "MINDS",
    "PrototypicalNetworks",
    "Evaluator",
    "EpisodicProtocol",
]

# Define a custom __getattr__ to import objects only when they're needed
def __getattr__(name):
    if name == "MINDS":
        from fewlearn.core.minds import MINDS as _MINDS
        return _MINDS
    elif name == "PrototypicalNetworks":
        from fewlearn.models.prototypical import PrototypicalNetworks as _PrototypicalNetworks
        return _PrototypicalNetworks
    elif name == "Evaluator":
        from fewlearn.evaluation.evaluator import Evaluator as _Evaluator
        return _Evaluator
    elif name == "EpisodicProtocol":
        from fewlearn.core.protocols import EpisodicProtocol as _EpisodicProtocol
        return _EpisodicProtocol
    else:
        raise AttributeError(f"module 'fewlearn' has no attribute '{name}'") 