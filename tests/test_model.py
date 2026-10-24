import pytest
import torch
from unittest.mock import patch, MagicMock
from src.models.efficientnet import create_model

@patch('src.models.efficientnet.efficientnet_b0')
def test_efficientnet_construction(mock_effnet):
    mock_net = MagicMock()
    mock_net.classifier = [None, MagicMock(in_features=1280)]
    mock_effnet.return_value = mock_net
    
    model = create_model(num_classes=1, pretrained=False)
    mock_effnet.assert_called_once()
    assert model is not None

def test_classifier_replacement():
    # Use a real model construction for test if possible, or mock
    model = create_model(num_classes=1, pretrained=False)
    # Check that output is 1
    assert model.classifier[1].out_features == 1

def test_output_shape():
    model = create_model(num_classes=1, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 1)

def test_cpu_inference():
    model = create_model(num_classes=1, pretrained=False)
    model.to('cpu')
    dummy_input = torch.randn(1, 3, 224, 224).to('cpu')
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    assert output.device.type == 'cpu'
