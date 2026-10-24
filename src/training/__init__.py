"""
Training module initialization.
"""

from .train import Trainer
from .metrics import (
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_roc_auc,
    compute_pr_auc,
    compute_confusion_matrix,
    classification_report_dict
)
from .evaluate import (
    evaluate_model,
    generate_predictions,
    compute_all_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_training_history
)
from .threshold import find_optimal_threshold, evaluate_at_threshold, threshold_analysis

__all__ = [
    "Trainer",
    "compute_accuracy",
    "compute_precision",
    "compute_recall",
    "compute_f1",
    "compute_roc_auc",
    "compute_pr_auc",
    "compute_confusion_matrix",
    "classification_report_dict",
    "evaluate_model",
    "generate_predictions",
    "compute_all_metrics",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_training_history",
    "find_optimal_threshold",
    "evaluate_at_threshold",
    "threshold_analysis"
]
