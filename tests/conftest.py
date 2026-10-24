import pytest
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def mock_config():
    return {
        "model": {"name": "efficientnet-b0", "num_classes": 1},
        "data": {"img_size": 224, "batch_size": 32},
        "training": {"epochs": 10}
    }
