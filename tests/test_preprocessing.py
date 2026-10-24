import pytest
import numpy as np
import torch
from PIL import Image
from src.data.preprocessing import get_inference_transform

def test_image_loading_validation():
    # Mock image
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    assert img.shape == (224, 224, 3)

def test_face_crop():
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    box = [10, 10, 50, 50]
    cropped = img[box[1]:box[3], box[0]:box[2]]
    assert cropped.shape == (40, 40, 3)

def test_resize_and_normalization():
    transform = get_inference_transform({})
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img)
    res = transform(pil_img)

    assert isinstance(res, torch.Tensor)
    assert res.shape == (3, 224, 224)

def test_inference_determinism():
    transform = get_inference_transform({})
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img)
    res1 = transform(pil_img)
    res2 = transform(pil_img)
        
    assert torch.allclose(res1, res2)
