"""
Core components of the FewLearn module.

This package contains the fundamental abstractions and interfaces
used throughout the FewLearn module.
"""

from .minds import MINDS
from .protocols import EpisodicProtocol

__all__ = ["MINDS", "EpisodicProtocol"] 