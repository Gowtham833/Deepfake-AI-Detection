"""
EfficientNet model definitions and utilities for deepfake detection.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

logger = logging.getLogger(__name__)

# Constants
REAL = 0
DEEPFAKE = 1
CLASS_NAMES = ["REAL", "DEEPFAKE"]

def create_model(config: Optional[Dict[str, Any]] = None, **kwargs) -> nn.Module:
    """
    Creates an EfficientNet-B0 model modified for single-class (binary) logit output.
    
    Args:
        config (Dict[str, Any], optional): Configuration dictionary containing model parameters.
            - pretrained (bool): Whether to use ImageNet weights. Default True.
            - dropout_rate (float): Dropout probability for the classifier. Default 0.2.
        **kwargs: Optional direct keyword arguments (pretrained, dropout_rate, num_classes).
            
    Returns:
        nn.Module: The modified EfficientNet-B0 model.
    """
    if config is None:
        config = {}
    pretrained = kwargs.get("pretrained", config.get("model", {}).get("pretrained", True))
    dropout_rate = kwargs.get("dropout_rate", config.get("model", {}).get("dropout_rate", 0.2))
    num_classes = kwargs.get("num_classes", config.get("model", {}).get("num_classes", 1))
    
    if pretrained:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
    else:
        model = efficientnet_b0(weights=None)
        
    # Replace classifier for binary classification (single logit)
    # The output logit will be passed to BCEWithLogitsLoss
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model

def freeze_backbone(model: nn.Module) -> None:
    """
    Freezes all layers in the EfficientNet backbone (features).
    
    Args:
        model (nn.Module): The EfficientNet model.
    """
    for param in model.features.parameters():
        param.requires_grad = False
    logger.info("Froze EfficientNet backbone.")

def unfreeze_backbone(model: nn.Module, from_layer: int = 0) -> None:
    """
    Unfreezes the backbone layers starting from a specific index.
    
    Args:
        model (nn.Module): The EfficientNet model.
        from_layer (int): The index of the layer in model.features to start unfreezing from.
    """
    features_len = len(model.features)
    from_layer = max(0, min(from_layer, features_len))
    
    for i in range(from_layer, features_len):
        for param in model.features[i].parameters():
            param.requires_grad = True
            
    logger.info(f"Unfroze EfficientNet backbone from layer {from_layer} to {features_len - 1}.")

def get_target_layer(model: nn.Module) -> nn.Module:
    """
    Gets the target layer for Grad-CAM (the last convolutional layer).
    
    Args:
        model (nn.Module): The EfficientNet model.
        
    Returns:
        nn.Module: The target layer.
    """
    return model.features[-1]

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    val_metric: float,
    config: Dict[str, Any],
    path: str
) -> None:
    """
    Saves a model checkpoint with full metadata.
    
    Args:
        model: The model to save.
        optimizer: The optimizer state to save.
        scheduler: The scheduler state to save (optional).
        epoch: Current epoch number.
        val_metric: Validation metric (e.g., validation loss or accuracy).
        config: Full configuration dictionary.
        path: File path to save the checkpoint.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "val_metric": val_metric,
        "config": config,
        "architecture": "efficientnet_b0",
        "class_mapping": {"REAL": REAL, "DEEPFAKE": DEEPFAKE},
        "input_size": config.get("data", {}).get("image_size", 224),
        "threshold": config.get("inference", {}).get("threshold", 0.5),
        "version": "1.0",
        "timestamp": datetime.now().isoformat()
    }
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved at {path}")

def load_checkpoint(
    path: str,
    model: nn.Module,
    device: torch.device,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Loads a checkpoint and its metadata.
    
    Args:
        path: Path to the checkpoint file.
        model: Model to load weights into.
        device: Device to map the tensors to.
        strict: Whether to strictly enforce that the keys in state_dict match.
        
    Returns:
        Dict[str, Any]: The loaded checkpoint dictionary containing metadata.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")
        
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    logger.info(f"Loaded checkpoint from {path} (Epoch: {checkpoint.get('epoch', 'N/A')}, Val Metric: {checkpoint.get('val_metric', 'N/A')})")
    return checkpoint
