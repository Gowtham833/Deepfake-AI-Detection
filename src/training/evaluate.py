"""
Evaluation and prediction functions for deepfake detection models.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import roc_curve, precision_recall_curve

from .metrics import (
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_roc_auc,
    compute_pr_auc,
    compute_confusion_matrix
)

logger = logging.getLogger(__name__)

# Constants
REAL = 0
DEEPFAKE = 1
CLASS_NAMES = ["REAL", "DEEPFAKE"]

def generate_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates predictions for an entire dataloader.
    
    Args:
        model: The trained model.
        dataloader: Dataloader containing validation/test data.
        device: Torch device (cpu or cuda).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Ground truth labels and predicted probabilities (for DEEPFAKE class).
    """
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).float()
            
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    return np.array(all_labels), np.array(all_probs)

def compute_all_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes a comprehensive set of classification metrics.
    
    Args:
        labels: Ground truth binary labels.
        probs: Predicted probabilities for the positive class (DEEPFAKE).
        threshold: Probability threshold for classification.
        
    Returns:
        Dict[str, Any]: Dictionary containing all computed metrics.
    """
    predictions = (probs >= threshold).astype(int)
    
    metrics = {
        "accuracy": compute_accuracy(labels, predictions),
        "precision": compute_precision(labels, predictions),
        "recall": compute_recall(labels, predictions),
        "f1": compute_f1(labels, predictions),
        "roc_auc": compute_roc_auc(labels, probs),
        "pr_auc": compute_pr_auc(labels, probs),
        "confusion_matrix": compute_confusion_matrix(labels, predictions).tolist()
    }
    
    return metrics

def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    End-to-end evaluation of a model on a given dataloader.
    
    Args:
        model: The trained model.
        dataloader: Evaluation dataloader.
        device: Torch device.
        threshold: Classification threshold.
        
    Returns:
        Dict[str, Any]: Dictionary of evaluation metrics.
    """
    labels, probs = generate_predictions(model, dataloader, device)
    metrics = compute_all_metrics(labels, probs, threshold)
    logger.info(f"Evaluation Results: Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, ROC AUC: {metrics['roc_auc']:.4f}")
    return metrics

def plot_confusion_matrix(cm: np.ndarray, save_path: str) -> None:
    """
    Plots and saves a confusion matrix.
    
    Args:
        cm: Confusion matrix array.
        save_path: Path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_roc_curve(labels: np.ndarray, probs: np.ndarray, save_path: str) -> None:
    """
    Plots and saves the ROC curve.
    
    Args:
        labels: Ground truth labels.
        probs: Predicted probabilities.
        save_path: Path to save the plot.
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = compute_roc_auc(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_pr_curve(labels: np.ndarray, probs: np.ndarray, save_path: str) -> None:
    """
    Plots and saves the Precision-Recall curve.
    
    Args:
        labels: Ground truth labels.
        probs: Predicted probabilities.
        save_path: Path to save the plot.
    """
    precision, recall, _ = precision_recall_curve(labels, probs)
    auc_score = compute_pr_auc(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {auc_score:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_training_history(history: Dict[str, list], save_path: str) -> None:
    """
    Plots training and validation loss and accuracy curves.
    
    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc' lists.
        save_path: Path to save the plot.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
