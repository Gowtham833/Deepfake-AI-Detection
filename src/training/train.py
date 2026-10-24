"""
Trainer class and training loop implementation.
"""

import logging
from typing import Dict, Any, Tuple
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.models.efficientnet import save_checkpoint
from .metrics import compute_accuracy

logger = logging.getLogger(__name__)

class Trainer:
    """
    Handles the training and validation loops for the model.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Training parameters from config
        train_cfg = self.config.get("training", {})
        self.lr = train_cfg.get("learning_rate", 1e-4)
        self.weight_decay = train_cfg.get("weight_decay", 1e-5)
        self.max_epochs = train_cfg.get("epochs", 10)
        self.patience = train_cfg.get("early_stopping_patience", 3)
        self.min_delta = train_cfg.get("early_stopping_min_delta", 1e-4)
        self.grad_clip = train_cfg.get("gradient_clipping", 1.0)
        
        # Paths
        self.checkpoint_dir = Path(self.config.get("paths", {}).get("checkpoints", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss, Optimizer, Scheduler
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        # Mixed precision
        self.use_amp = train_cfg.get("mixed_precision", True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # History
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        all_labels = []
        all_preds = []
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device).float()
            
            self.optimizer.zero_grad()
            
            if self.use_amp and self.scaler is not None:
                with autocast():
                    logits = self.model(images).squeeze(1)
                    loss = self.criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images).squeeze(1)
                loss = self.criterion(logits, labels)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                
            total_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(self.train_loader.dataset)
        acc = compute_accuracy(all_labels, all_preds)
        return avg_loss, acc

    def validate_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device).float()
                
                if self.use_amp and self.scaler is not None:
                    with autocast():
                        logits = self.model(images).squeeze(1)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(images).squeeze(1)
                    loss = self.criterion(logits, labels)
                    
                total_loss += loss.item() * images.size(0)
                
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                
        avg_loss = total_loss / len(self.val_loader.dataset)
        acc = compute_accuracy(all_labels, all_preds)
        return avg_loss, acc

    def fit(self) -> Dict[str, list]:
        """
        Executes the full training loop with early stopping and checkpointing.
        Returns the training history.
        """
        logger.info(f"Starting training for {self.max_epochs} epochs...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, self.max_epochs + 1):
            logger.info(f"Epoch {epoch}/{self.max_epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping and checkpointing
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                checkpoint_path = str(self.checkpoint_dir / "best_model.pth")
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    val_metric=best_val_loss,
                    config=self.config,
                    path=checkpoint_path
                )
                logger.info(f"Saved new best model with val_loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                logger.info(f"Early stopping counter: {patience_counter} out of {self.patience}")
                if patience_counter >= self.patience:
                    logger.info("Early stopping triggered.")
                    break
                    
        return self.history
