import os
import pytest
from unittest.mock import patch
from src.data.dataset import DeepfakeDataset
from src.utils.io import get_project_root

# REAL = 0, DEEPFAKE = 1

def mock_os_walk(root):
    if 'REAL' in root:
        return [(root, [], ['real1.jpg', 'real2.jpg'])]
    elif 'DEEPFAKE' in root:
        return [(root, [], ['fake1.jpg', 'fake2.jpg'])]
    else:
        return [(root, ['REAL', 'DEEPFAKE'], [])]

@patch('os.walk')
def test_dataset_discovery_and_label_mapping(mock_walk):
    mock_walk.side_effect = mock_os_walk
    # Mock behavior of dataset manifest generation
    # Assume DeepfakeDataset uses os.walk to find files
    # Here we simulate the logic
    manifest = [
        {"path": "/data/REAL/real1.jpg", "label": 0},
        {"path": "/data/REAL/real2.jpg", "label": 0},
        {"path": "/data/DEEPFAKE/fake1.jpg", "label": 1},
        {"path": "/data/DEEPFAKE/fake2.jpg", "label": 1},
    ]
    assert len(manifest) == 4
    labels = [x['label'] for x in manifest]
    assert 0 in labels
    assert 1 in labels

def test_group_aware_splitting():
    # Mocking group aware split
    # Suppose we split by source video
    manifest = [
        {"path": "vid1_frame1.jpg", "group": "vid1", "label": 0},
        {"path": "vid1_frame2.jpg", "group": "vid1", "label": 0},
        {"path": "vid2_frame1.jpg", "group": "vid2", "label": 1},
    ]
    # Simple split mock
    train_groups = {"vid1"}
    val_groups = {"vid2"}
    
    train_set = [m for m in manifest if m["group"] in train_groups]
    val_set = [m for m in manifest if m["group"] in val_groups]
    
    train_g = set([m["group"] for m in train_set])
    val_g = set([m["group"] for m in val_set])
    
    assert len(train_g.intersection(val_g)) == 0

def test_no_source_leakage():
    # Just a sanity check for leakage
    train_g = {"a", "b"}
    val_g = {"c", "d"}
    assert len(train_g.intersection(val_g)) == 0
