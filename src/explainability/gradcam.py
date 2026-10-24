"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for deepfake visual explainability.
"""

import logging
from typing import Dict, Optional, Any
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class GradCAM:
    """
    Computes Grad-CAM activations for a given target layer in a model.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hooks = []
        
        self._register_hooks()
        
    def _register_hooks(self) -> None:
        """Registers forward and backward hooks to capture activations and gradients."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def full_backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
            
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(full_backward_hook))
        
    def remove_hooks(self) -> None:
        """Removes the registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
            
    def __del__(self):
        self.remove_hooks()
        
    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generates the Grad-CAM heatmap for the input tensor.
        
        Args:
            input_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W).
            target_class (Optional[int]): Target class to compute the heatmap for. 
                                          If None, uses the model's predicted class.
                                          Note: For single logit binary classification, this doesn't explicitly
                                          select a class index but we backward() on the raw output logit.
                                          
        Returns:
            np.ndarray: The normalized heatmap (H, W) in [0, 1].
        """
        self.model.eval()
        # Enable gradients for the input for visualization purposes
        input_tensor.requires_grad_(True)
        
        self.model.zero_grad()
        output_logits = self.model(input_tensor)
        
        # For single-output binary classification, we just compute gradients w.r.t the only logit
        # If target_class is specified and it's 0 (REAL) in a DEEPFAKE-positive model, we negate the logit
        # to find activations supporting the REAL class, although standard practice is just backwarding the logit.
        score = output_logits.squeeze()
        if target_class == 0:
            score = -score
            
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            raise ValueError("Forward/backward hooks did not capture activations/gradients.")
            
        # Global average pooling on the gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        
        # Apply ReLU to only keep positive influences
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
            
        return cam.cpu().numpy()

def apply_colormap(heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Applies a color map to a normalized heatmap [0, 1].
    
    Args:
        heatmap: Normalized heatmap in [0, 1].
        colormap: OpenCV colormap constant.
        
    Returns:
        np.ndarray: Colored heatmap in BGR format, uint8.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    return colored_heatmap

def create_overlay(original_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blends the original RGB image with the colored heatmap.
    
    Args:
        original_image: Original RGB image array (H, W, 3), uint8.
        heatmap: Colored heatmap (BGR from cv2 or RGB) array (H, W, 3), uint8.
        alpha: Blend weight for the heatmap.
        
    Returns:
        np.ndarray: Blended RGB image, uint8.
    """
    # Resize heatmap to match original image dimensions if necessary
    if heatmap.shape[:2] != original_image.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
    # Convert heatmap BGR to RGB if assuming it's output of apply_colormap
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_rgb, alpha, 0)
    return overlay

def get_target_layer_from_model(model: nn.Module, layer_name: str = 'auto') -> nn.Module:
    """
    Gets the target layer for Grad-CAM.
    
    Args:
        model: The PyTorch model.
        layer_name: The name of the layer or 'auto' to pick the last feature layer.
        
    Returns:
        nn.Module: The requested layer.
    """
    if layer_name == 'auto':
        # Default for EfficientNet
        if hasattr(model, 'features'):
            return model.features[-1]
        raise ValueError("Cannot automatically determine target layer. Please provide an explicit layer_name.")
    else:
        # Traverse model structure based on dot-separated names
        module = model
        for name in layer_name.split('.'):
            module = getattr(module, name)
        return module

def generate_gradcam_visualizations(
    model: nn.Module,
    input_tensor: torch.Tensor,
    original_face_image: np.ndarray,
    config: Dict[str, Any],
    device: torch.device
) -> Dict[str, np.ndarray]:
    """
    Generates the complete Grad-CAM pipeline visualization.
    
    Args:
        model: The PyTorch model.
        input_tensor: Preprocessed input tensor (1, C, H, W) moved to device.
        original_face_image: Original RGB image as numpy array (H, W, 3).
        config: App configuration dictionary.
        device: Torch device.
        
    Returns:
        Dict[str, np.ndarray]: Dictionary containing 'heatmap' (1D normalized), 
                               'colored_heatmap' (BGR), and 'overlay' (RGB).
    """
    target_layer = get_target_layer_from_model(model, 'auto')
    
    gradcam = GradCAM(model, target_layer)
    try:
        heatmap = gradcam.generate(input_tensor.to(device))
        colored_heatmap = apply_colormap(heatmap)
        overlay = create_overlay(original_face_image, colored_heatmap, alpha=0.5)
    finally:
        gradcam.remove_hooks()
        
    return {
        "heatmap": heatmap,
        "colored_heatmap": colored_heatmap,
        "overlay": overlay
    }
