"""
Evaluation components for few-shot learning.

This package contains tools for evaluating few-shot learning models,
computing metrics, and analyzing results.
"""

from evaluation.metrics import calculate_metrics, accuracy, f1_score
from evaluation.evaluator import Evaluator

__all__ = ["Evaluator", "calculate_metrics", "accuracy", "f1_score"] 