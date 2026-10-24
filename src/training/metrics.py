"""
Metrics calculations for deepfake classification using sklearn.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Any, List

# Constants
REAL = 0
DEEPFAKE = 1
CLASS_NAMES = ["REAL", "DEEPFAKE"]

def compute_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Computes overall accuracy."""
    return accuracy_score(labels, predictions)

def compute_precision(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Computes precision for the positive class (DEEPFAKE)."""
    return precision_score(labels, predictions, pos_label=DEEPFAKE, zero_division=0)

def compute_recall(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Computes recall for the positive class (DEEPFAKE)."""
    return recall_score(labels, predictions, pos_label=DEEPFAKE, zero_division=0)

def compute_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Computes F1 score for the positive class (DEEPFAKE)."""
    return f1_score(labels, predictions, pos_label=DEEPFAKE, zero_division=0)

def compute_roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Computes Area Under the Receiver Operating Characteristic Curve."""
    if len(np.unique(labels)) < 2:
        return 0.5 # Default when only one class is present
    return roc_auc_score(labels, probs)

def compute_pr_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Computes Area Under the Precision-Recall Curve (Average Precision)."""
    if len(np.unique(labels)) < 2:
        return 0.0 # Default when only one class is present
    return average_precision_score(labels, probs)

def compute_confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Computes the confusion matrix."""
    return confusion_matrix(labels, predictions, labels=[REAL, DEEPFAKE])

def classification_report_dict(labels: np.ndarray, predictions: np.ndarray, target_names: List[str] = CLASS_NAMES) -> Dict[str, Any]:
    """Generates a detailed classification report as a dictionary."""
    return classification_report(labels, predictions, target_names=target_names, output_dict=True, zero_division=0)
