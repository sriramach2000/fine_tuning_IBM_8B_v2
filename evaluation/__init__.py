"""
Evaluation Module for Iterative Teacher-Student Distillation

This module provides quality evaluation for student model outputs
to determine which need teacher correction.
"""

from .code_quality_metrics import CodeQualityEvaluator, QualityScore

__all__ = ['CodeQualityEvaluator', 'QualityScore']
