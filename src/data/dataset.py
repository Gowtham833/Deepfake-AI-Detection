import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import logging
from PIL import Image

logger = logging.getLogger(__name__)

REAL = 0
DEEPFAKE = 1
class_names = ["REAL", "DEEPFAKE"]

class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for deepfake detection.
    """
    def __init__(self, manifest_df, transform=None, config=None):
        self.manifest_df = manifest_df.reset_index(drop=True)
        self.transform = transform
        self.config = config

    def __len__(self):
        return len(self.manifest_df)

    def __getitem__(self, idx):
        row = self.manifest_df.iloc[idx]
        image_path = row['image_path']
        label = row['label']

        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image_tensor = self.transform(image)
            else:
                import torchvision.transforms as T
                image_tensor = T.ToTensor()(image)
            return image_tensor, label
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            return None

def custom_collate_fn(batch):
    """Filters out None samples before collating."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)

def create_dataloaders(train_df, val_df, test_df, config):
    from .preprocessing import get_training_transform, get_inference_transform
    train_transform = get_training_transform(config)
    eval_transform = get_inference_transform(config)

    train_dataset = DeepfakeDataset(train_df, transform=train_transform, config=config)
    val_dataset = DeepfakeDataset(val_df, transform=eval_transform, config=config)
    
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=custom_collate_fn)
    
    test_loader = None
    if test_df is not None and len(test_df) > 0:
        test_dataset = DeepfakeDataset(test_df, transform=eval_transform, config=config)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                 num_workers=num_workers, collate_fn=custom_collate_fn)

    return train_loader, val_loader, test_loader
