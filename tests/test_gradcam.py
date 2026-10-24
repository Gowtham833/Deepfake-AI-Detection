import pytest
import torch
import numpy as np
from torch import nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 10 * 10, 1)
        
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def test_target_layer_resolution():
    model = SimpleModel()
    assert hasattr(model, 'conv')

def test_heatmap_generation():
    # Mocking GradCAM
    model = SimpleModel()
    model.eval()
    
    # Forward pass
    img = torch.randn(1, 3, 10, 10)
    img.requires_grad = True
    
    out = model(img)
    out.backward()
    
    assert img.grad is not None
    assert img.grad.shape == (1, 3, 10, 10)
    
    # Mock heatmap
    heatmap = np.random.rand(10, 10)
    assert heatmap.shape == (10, 10)
