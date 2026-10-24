"""
Threshold optimization utilities for finding the best classification boundary.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.metrics import confusion_matrix

from .metrics import (
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_f1
)

def evaluate_at_threshold(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    """
    Evaluates basic metrics at a specific decision threshold.
    
    Args:
        labels: Ground truth binary labels.
        probs: Predicted probabilities.
        threshold: Classification threshold.
        
    Returns:
        Dict[str, float]: Metrics evaluated at the given threshold.
    """
    preds = (probs >= threshold).astype(int)
    
    accuracy = compute_accuracy(labels, preds)
    precision = compute_precision(labels, preds)
    recall = compute_recall(labels, preds)
    f1 = compute_f1(labels, preds)
    
    cm = confusion_matrix(labels, preds)
    # Handle edge case where only one class is predicted
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0
        
    balanced_accuracy = (recall + specificity) / 2.0
    youden_j = recall + specificity - 1.0
    
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "youden": youden_j
    }

def threshold_analysis(labels: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    """
    Performs evaluation across thresholds from 0.01 to 0.99.
    
    Args:
        labels: Ground truth labels.
        probs: Predicted probabilities.
        
    Returns:
        pd.DataFrame: DataFrame containing metrics at each threshold.
    """
    results = []
    thresholds = np.arange(0.01, 1.00, 0.01)
    
    for t in thresholds:
        metrics = evaluate_at_threshold(labels, probs, t)
        results.append(metrics)
        
    return pd.DataFrame(results)

def find_optimal_threshold(labels: np.ndarray, probs: np.ndarray, method: str = 'f1') -> Tuple[float, float]:
    """
    Finds the optimal threshold based on a specific optimization target.
    
    Args:
        labels: Ground truth labels.
        probs: Predicted probabilities.
        method: Method for optimization ('f1', 'balanced_accuracy', or 'youden').
        
    Returns:
        Tuple[float, float]: Optimal threshold and its corresponding best score.
    """
    df = threshold_analysis(labels, probs)
    
    if method not in ['f1', 'balanced_accuracy', 'youden']:
        raise ValueError(f"Unknown optimization method: {method}")
        
    best_idx = df[method].idxmax()
    best_row = df.loc[best_idx]
    
    return float(best_row['threshold']), float(best_row[method])
