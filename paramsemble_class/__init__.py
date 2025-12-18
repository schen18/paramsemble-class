"""
Paramsemble-Class - Ensemble Logistic Regression

A scikit-learn compatible library for advanced classification using
ensemble methods based on combinatorial feature selection.
"""

__version__ = "0.5.0"

from paramsemble_class.core.elr_classifier import ELRClassifier
from paramsemble_class.scoring.scorer import ModelScorer
from paramsemble_class.sql.generator import SQLGenerator

__all__ = [
    "ELRClassifier",
    "ModelScorer",
    "SQLGenerator",
]
