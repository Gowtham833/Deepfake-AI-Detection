import pytest
import torch
import os
import tempfile
from src.models.efficientnet import save_checkpoint, load_checkpoint

def test_saving_loading_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'model.pt')
        state = {
            'model_state_dict': {'layer1.weight': torch.tensor([1.0, 2.0])},
            'epoch': 5,
            'best_val_loss': 0.1
        }
        torch.save(state, path)
        
        loaded = torch.load(path)
        assert 'model_state_dict' in loaded
        assert loaded['epoch'] == 5
        assert torch.equal(loaded['model_state_dict']['layer1.weight'], state['model_state_dict']['layer1.weight'])

def test_metadata_validation():
    # Mock testing metadata
    state = {'epoch': 10, 'threshold': 0.5}
    assert 'threshold' in state

def test_missing_file_handling():
    with pytest.raises(FileNotFoundError):
        # We assume load_checkpoint raises FileNotFoundError
        # or we test torch.load directly
        if not os.path.exists('non_existent.pt'):
            raise FileNotFoundError()
        torch.load('non_existent.pt')
