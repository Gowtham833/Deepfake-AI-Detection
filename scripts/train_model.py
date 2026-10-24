#!/usr/bin/env python3
"""
Train deepfake detection model on image dataset.

Usage:
    python scripts/train_model.py --train-root "C:/Users/Dell/Downloads/archive/train-20250112T065955Z-001/train" --val-root "C:/Users/Dell/Downloads/archive/test-20250112T065939Z-001/test" --epochs 20

Dataset expected structure:
    <data-root>/
        real/
            *.jpg
        fake/
            *.jpg
"""
import sys
import os
from pathlib import Path
import argparse
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, classification_report

from src.models.efficientnet import create_model, save_checkpoint
from src.utils.seed import set_seed
from src.utils.device import get_device

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Simple folder dataset  (no face-detection preprocessing during training)
# ─────────────────────────────────────────────────────────────────────────────
class ImageFolderDataset(torch.utils.data.Dataset):
    """Loads images from real/ and fake/ sub-folders."""

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples = []   # list of (path, label)

        real_dir = os.path.join(root, 'real')
        fake_dir = os.path.join(root, 'fake')

        for d, label in [(real_dir, 0), (fake_dir, 1)]:
            if not os.path.isdir(d):
                logger.warning(f"Directory not found: {d}")
                continue
            for fname in sorted(os.listdir(d)):
                if Path(fname).suffix.lower() in self.EXTENSIONS:
                    self.samples.append((os.path.join(d, fname), label))

        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        logger.info(f"Dataset '{root}': {n_real} real, {n_fake} fake  (total {len(self.samples)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


def make_weighted_sampler(dataset: ImageFolderDataset):
    """WeightedRandomSampler that up-samples the minority class."""
    labels = [s[1] for s in dataset.samples]
    counts = np.bincount(labels)
    class_weights = 1.0 / counts.astype(float)
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def get_transforms(image_size: int = 224):
    """Return (train_transform, val_transform)."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tfm = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_tfm, val_tfm


# ─────────────────────────────────────────────────────────────────────────────
# Training / validation helpers
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss, all_probs, all_labels = 0.0, [], []

    pbar = tqdm(loader, desc='  Train', leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_amp:
            with autocast():
                logits = model(imgs).squeeze(1)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    avg_loss = total_loss / len(loader.dataset)
    preds = (np.array(all_probs) >= 0.5).astype(int)
    acc = (preds == np.array(all_labels)).mean()
    return avg_loss, float(acc)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []

    for imgs, labels in tqdm(loader, desc='  Val  ', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        if use_amp:
            with autocast():
                logits = model(imgs).squeeze(1)
                loss = criterion(logits, labels)
        else:
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, labels)

        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item() * imgs.size(0)

    avg_loss = total_loss / len(loader.dataset)
    probs_arr  = np.array(all_probs)
    labels_arr = np.array(all_labels)
    preds = (probs_arr >= 0.5).astype(int)
    acc = (preds == labels_arr).mean()
    try:
        auc = roc_auc_score(labels_arr, probs_arr)
    except Exception:
        auc = 0.0
    f1 = f1_score(labels_arr, preds, zero_division=0)
    return avg_loss, float(acc), float(auc), float(f1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument(
        "--train-root", type=str,
        default=r"C:\Users\Dell\Downloads\archive\train-20250112T065955Z-001\train",
        help="Root training dir (contains real/ and fake/)")
    parser.add_argument(
        "--val-root", type=str,
        default=r"C:\Users\Dell\Downloads\archive\test-20250112T065939Z-001\test",
        help="Root validation dir (contains real/ and fake/)")
    parser.add_argument("--checkpoint-dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    logger.info(f"Device: {device}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_tfm, val_tfm = get_transforms(args.image_size)
    train_ds = ImageFolderDataset(args.train_root, transform=train_tfm)
    val_ds   = ImageFolderDataset(args.val_root,   transform=val_tfm)

    if len(train_ds) == 0:
        logger.error("Training dataset empty. Check --train-root."); sys.exit(1)

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    # Pos-weight for loss (handles class imbalance)
    labels_list = [s[1] for s in train_ds.samples]
    n_real = sum(1 for l in labels_list if l == 0)
    n_fake = sum(1 for l in labels_list if l == 1)
    pos_weight = torch.tensor([n_real / max(n_fake, 1)], device=device)
    logger.info(f"pos_weight = {pos_weight.item():.3f}")

    config   = {"model": {"pretrained": True, "dropout_rate": 0.4}}
    model    = create_model(config).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    use_amp = not args.no_amp and torch.cuda.is_available()
    scaler  = GradScaler() if use_amp else None

    # ── Stage 1: frozen backbone ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 1: Training classifier head (backbone frozen)")
    logger.info("=" * 60)

    for p in model.features.parameters():
        p.requires_grad = False

    opt1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4)
    sch1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt1, mode='min', factor=0.5, patience=2)

    best_loss = float('inf')
    wait = 0
    s1_epochs = min(args.epochs // 2, 10)

    for ep in range(1, s1_epochs + 1):
        logger.info(f"  Epoch {ep}/{s1_epochs}")
        tl, ta = train_one_epoch(model, train_loader, criterion, opt1, scaler, device, use_amp)
        vl, va, vauc, vf1 = evaluate(model, val_loader, criterion, device, use_amp)
        sch1.step(vl)
        logger.info(f"    train loss={tl:.4f} acc={ta:.4f} | "
                    f"val loss={vl:.4f} acc={va:.4f} AUC={vauc:.4f} F1={vf1:.4f}")

        if vl < best_loss - 1e-4:
            best_loss = vl; wait = 0
            save_checkpoint(model, opt1, sch1, ep, best_loss,
                            {"val_loss": best_loss, "val_auc": vauc, "val_f1": vf1},
                            str(ckpt_dir / "best_model.pth"))
            logger.info(f"    ✓ New best saved (val_loss={best_loss:.4f})")
        else:
            wait += 1
            if wait >= 3:
                logger.info("    Early stopping – Stage 1"); break

    # ── Stage 2: full fine-tune ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 2: Fine-tuning full network (backbone unfrozen)")
    logger.info("=" * 60)

    for p in model.features.parameters():
        p.requires_grad = True

    # Reload best Stage-1 weights
    ckpt_data = torch.load(str(ckpt_dir / "best_model.pth"), map_location=device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    logger.info("  Reloaded Stage-1 best weights")

    s2_epochs = args.epochs - s1_epochs
    opt2 = torch.optim.AdamW(model.parameters(), lr=args.lr / 10, weight_decay=1e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=s2_epochs)

    best_loss2 = float('inf')
    wait2 = 0

    for ep in range(1, s2_epochs + 1):
        logger.info(f"  Epoch {ep}/{s2_epochs}")
        tl, ta = train_one_epoch(model, train_loader, criterion, opt2, scaler, device, use_amp)
        vl, va, vauc, vf1 = evaluate(model, val_loader, criterion, device, use_amp)
        sch2.step()
        logger.info(f"    train loss={tl:.4f} acc={ta:.4f} | "
                    f"val loss={vl:.4f} acc={va:.4f} AUC={vauc:.4f} F1={vf1:.4f}")

        if vl < best_loss2 - 1e-4:
            best_loss2 = vl; wait2 = 0
            save_checkpoint(model, opt2, sch2, ep, best_loss2,
                            {"val_loss": best_loss2, "val_auc": vauc, "val_f1": vf1},
                            str(ckpt_dir / "best_model.pth"))
            logger.info(f"    ✓ New best saved (val_loss={best_loss2:.4f})")
        else:
            wait2 += 1
            if wait2 >= 5:
                logger.info("    Early stopping – Stage 2"); break

    # ── Final report ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("FINAL Evaluation on Validation Set")
    logger.info("=" * 60)
    ckpt_data = torch.load(str(ckpt_dir / "best_model.pth"), map_location=device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labs in val_loader:
            logits = model(imgs.to(device)).squeeze(1)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labs.numpy())

    preds = (np.array(all_probs) >= 0.5).astype(int)
    logger.info("\n" + classification_report(
        all_labels, preds, target_names=['REAL', 'FAKE'], zero_division=0))
    try:
        logger.info(f"ROC-AUC: {roc_auc_score(all_labels, all_probs):.4f}")
    except Exception:
        pass
    logger.info(f"\nDone! Best model → {ckpt_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
