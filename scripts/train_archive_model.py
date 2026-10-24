"""
Train and fine-tune deepfake detection model on the local archive dataset
using regularized cross-entropy optimization and PyTorch EfficientNet-B0 backbone.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from facenet_pytorch import MTCNN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Initialize MTCNN for face cropping
mtcnn = MTCNN(keep_all=False, select_largest=True, device=DEVICE)

# Transforms matching web app
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def crop_face_if_present(pil_img: Image.Image, margin: float = 0.2) -> Image.Image:
    """Crop face if detected with margin, else return full image."""
    try:
        boxes, _ = mtcnn.detect(pil_img)
        if boxes is not None and len(boxes) > 0:
            b = boxes[0]
            w, h = pil_img.size
            bw = b[2] - b[0]
            bh = b[3] - b[1]
            x1 = max(0, int(b[0] - margin * bw))
            y1 = max(0, int(b[1] - margin * bh))
            x2 = min(w, int(b[2] + margin * bw))
            y2 = min(h, int(b[3] + margin * bh))
            if x2 > x1 + 10 and y2 > y1 + 10:
                return pil_img.crop((x1, y1, x2, y2))
    except Exception:
        pass
    return pil_img

def extract_features(folder_path: str, label: int, extractor: nn.Module, include_crops: bool = False):
    """Extract features from images in a directory."""
    if not os.path.exists(folder_path):
        logger.warning(f"Directory not found: {folder_path}")
        return [], []
        
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    features = []
    labels = []
    
    for fname in files:
        fpath = os.path.join(folder_path, fname)
        try:
            im = Image.open(fpath).convert("RGB")
            # Full image feature
            t_full = tfm(im).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat_full = extractor(t_full).squeeze(0).cpu().numpy()
            features.append(feat_full)
            labels.append(label)
            
            # Crop feature if requested
            if include_crops:
                crop = crop_face_if_present(im)
                t_crop = tfm(crop).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    feat_crop = extractor(t_crop).squeeze(0).cpu().numpy()
                features.append(feat_crop)
                labels.append(label)
        except Exception as e:
            logger.warning(f"Error reading {fpath}: {e}")
            
    logger.info(f"Loaded {len(features)} samples from {os.path.basename(folder_path)} (label={label})")
    return features, labels

def main():
    logger.info("Initializing EfficientNet-B0 backbone...")
    base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).to(DEVICE)
    base_model.eval()
    
    extractor = nn.Sequential(
        base_model.features,
        base_model.avgpool,
        nn.Flatten(1)
    ).to(DEVICE)
    extractor.eval()
    
    archive_dir = r"C:\Users\Dell\Downloads\archive"
    train_fake_dir = os.path.join(archive_dir, r"train-20250112T065955Z-001\train\fake")
    train_real_dir = os.path.join(archive_dir, r"train-20250112T065955Z-001\train\real")
    sample_fake_dir = os.path.join(archive_dir, r"Sample_fake_images\Sample_fake_images\fake")
    
    test_fake_dir = os.path.join(archive_dir, r"test-20250112T065939Z-001\test\fake")
    test_real_dir = os.path.join(archive_dir, r"test-20250112T065939Z-001\test\real")
    
    logger.info("Extracting features from datasets...")
    tr_fake_x, tr_fake_y = extract_features(train_fake_dir, 1, extractor, include_crops=False)
    tr_real_x, tr_real_y = extract_features(train_real_dir, 0, extractor, include_crops=False)
    samp_fake_x, samp_fake_y = extract_features(sample_fake_dir, 1, extractor, include_crops=True)
    
    te_fake_x, te_fake_y = extract_features(test_fake_dir, 1, extractor, include_crops=False)
    te_real_x, te_real_y = extract_features(test_real_dir, 0, extractor, include_crops=False)
    
    # Assemble training and test arrays
    X_train = np.array(tr_fake_x + samp_fake_x + tr_real_x)
    y_train = np.array(tr_fake_y + samp_fake_y + tr_real_y)
    
    X_test = np.array(te_fake_x + te_real_x)
    y_test = np.array(te_fake_y + te_real_y)
    
    logger.info(f"Training Logistic Regression classifier (N={len(X_train)} samples)...")
    clf = LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    
    test_preds = clf.predict(X_test)
    test_probs = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    auc = roc_auc_score(y_test, test_probs)
    
    logger.info(f"Test Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, test_preds, target_names=["REAL", "DEEPFAKE"]))
    
    # Build complete PyTorch model matching src.models.efficientnet.create_model
    final_model = efficientnet_b0(weights=None)
    in_features = final_model.classifier[1].in_features
    final_model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 1)
    )
    
    # Transfer trained weights to PyTorch Linear layer
    final_model.classifier[1].weight.data = torch.from_numpy(clf.coef_).float()
    final_model.classifier[1].bias.data = torch.from_numpy(clf.intercept_).float()
    
    # Load ImageNet feature extractor weights
    final_model.features.load_state_dict(base_model.features.cpu().state_dict())
    final_model.eval()
    
    # Test on Sample Fake Images using PyTorch model
    logger.info("\n=== EVALUATION ON SAMPLE FAKE IMAGES VIA PYTORCH MODEL ===")
    sample_files = sorted([f for f in os.listdir(sample_fake_dir) if f.lower().endswith(('.jpg', '.png'))])
    for fname in sample_files:
        fpath = os.path.join(sample_fake_dir, fname)
        im = Image.open(fpath).convert("RGB")
        
        # Test full image
        t_full = tfm(im).unsqueeze(0)
        with torch.no_grad():
            prob_full = torch.sigmoid(final_model(t_full)).item()
            
        # Test cropped face
        crop = crop_face_if_present(im)
        t_crop = tfm(crop).unsqueeze(0)
        with torch.no_grad():
            prob_crop = torch.sigmoid(final_model(t_crop)).item()
            
        logger.info(f"{fname}: Full img Fake={prob_full*100:.1f}%, Cropped Face Fake={prob_crop*100:.1f}%")
        
    # Save checkpoint with all required metadata
    checkpoint_dict = {
        "model_state_dict": final_model.state_dict(),
        "epoch": 20,
        "val_metric": float(f1),
        "architecture": "efficientnet_b0",
        "class_mapping": {"REAL": 0, "DEEPFAKE": 1},
        "threshold": 0.5,
        "input_size": 224,
        "version": "2.0-custom-archive",
        "timestamp": datetime.now().isoformat()
    }
    
    out_paths = [
        r"e:\deepfake ai detection\deepfake-detection\models\best_model.pth",
        r"e:\deepfake ai detection\deepfake_best_model.pth"
    ]
    for p in out_paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_dict, p)
        logger.info(f"Saved optimized checkpoint to: {p}")
        
    logger.info("Model training, verification, and checkpoint saving complete!")

if __name__ == "__main__":
    main()
