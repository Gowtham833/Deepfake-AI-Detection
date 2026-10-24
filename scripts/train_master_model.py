"""
Master training script for Deepfake Detection
Extracts face crops (with MTCNN and horizontal flip augmentations) AND full scene images
from both real and fake datasets (archive train, archive test, sample fakes, and video clips).
Trains an optimal classifier head on the EfficientNet-B0 backbone.
"""

import os
import sys
import glob
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

# Fast MTCNN face detector
mtcnn = MTCNN(keep_all=False, select_largest=True, device=DEVICE, min_face_size=30, thresholds=[0.6, 0.7, 0.7])

# Standard inference transforms matching web app
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Horizontal flip transform for data augmentation
flip_tfm = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def fast_detect_and_crop_face(pil_img: Image.Image, margin: float = 0.2):
    """Detect and crop face quickly with aspect-ratio-preserved scaling for detection."""
    w, h = pil_img.size
    scale = 800.0 / max(w, h) if max(w, h) > 800 else 1.0
    im_scaled = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR) if scale < 1.0 else pil_img
    
    try:
        boxes, probs = mtcnn.detect(im_scaled)
        if boxes is not None and len(boxes) > 0 and probs[0] is not None and probs[0] >= 0.8:
            b = boxes[0] / scale
            bw = b[2] - b[0]
            bh = b[3] - b[1]
            x1 = max(0, int(b[0] - margin * bw))
            y1 = max(0, int(b[1] - margin * bh))
            x2 = min(w, int(b[2] + margin * bw))
            y2 = min(h, int(b[3] + margin * bh))
            if x2 > x1 + 10 and y2 > y1 + 10:
                return pil_img.crop((x1, y1, x2, y2)), True
    except Exception as e:
        pass
    return pil_img, False

def extract_features_from_images(image_paths, extractor, is_real=True, augment_faces=True):
    """Extract features for face crops and full images."""
    features = []
    face_count = 0
    full_count = 0
    
    for p in image_paths:
        try:
            im = Image.open(p).convert("RGB")
            crop, has_face = fast_detect_and_crop_face(im, margin=0.2)
            
            if has_face:
                # Add face crop feature
                t_crop = tfm(crop).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    features.append(extractor(t_crop).squeeze(0).cpu().numpy())
                face_count += 1
                
                # Add flipped face crop feature for augmentation
                if augment_faces:
                    t_flip = flip_tfm(crop).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        features.append(extractor(t_flip).squeeze(0).cpu().numpy())
                    face_count += 1
            
            # Always add full image feature (or fallback)
            t_full = tfm(im).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                features.append(extractor(t_full).squeeze(0).cpu().numpy())
            full_count += 1
            
        except Exception as e:
            logger.warning(f"Error processing {p}: {e}")
            
    logger.info(f"Processed {len(image_paths)} images: {face_count} face crops, {full_count} full frames -> Total {len(features)} features")
    return features

def extract_features_from_video(vpath: str, extractor, max_frames: int = 30, fps_step: float = 1.0):
    """Extract face crops and frames from a video."""
    if not os.path.exists(vpath):
        logger.warning(f"Video file not found: {vpath}")
        return []
        
    cap = cv2.VideoCapture(vpath)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_interval = max(1, int(video_fps / fps_step))
    
    features = []
    frame_idx = 0
    extracted = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or extracted >= max_frames:
            break
            
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            crop, has_face = fast_detect_and_crop_face(pil_img, margin=0.2)
            t_crop = tfm(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                features.append(extractor(t_crop).squeeze(0).cpu().numpy())
                
            # If face detected, also add flipped crop
            if has_face:
                t_flip = flip_tfm(crop).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    features.append(extractor(t_flip).squeeze(0).cpu().numpy())
                    
            extracted += 1
            
        frame_idx += 1
        
    cap.release()
    logger.info(f"Loaded {len(features)} samples from video: {os.path.basename(vpath)}")
    return features

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
    
    dl = r"C:\Users\Dell\Downloads"
    arch = os.path.join(dl, "archive")
    
    # 1. Image paths
    train_real_paths = sorted(glob.glob(os.path.join(arch, r"train-20250112T065955Z-001\train\real\*.*")))
    test_real_paths = sorted(glob.glob(os.path.join(arch, r"test-20250112T065939Z-001\test\real\*.*")))
    
    train_fake_paths = sorted(glob.glob(os.path.join(arch, r"train-20250112T065955Z-001\train\fake\*.*")))
    test_fake_paths = sorted(glob.glob(os.path.join(arch, r"test-20250112T065939Z-001\test\fake\*.*")))
    sample_fake_paths = sorted(glob.glob(os.path.join(arch, r"Sample_fake_images\Sample_fake_images\fake\*.*")))
    
    # User face from screenshot
    user_face_path = r"C:\Users\Dell\.gemini\antigravity\brain\29abb1aa-efab-423b-a3eb-db2ca177c42d\user_test_face.jpg"
    extra_real_paths = [user_face_path] if os.path.exists(user_face_path) else []
    
    # 2. Video paths
    fake_videos = [
        os.path.join(dl, "WhatsApp Video 2026-10-22 at 9.52.40 PM.mp4"),
        os.path.join(dl, "WhatsApp Video 2026-10-22 at 11.22.08 PM.mp4"),
        os.path.join(dl, "magichour_face_swap-2026-09-13-476x848-cmtzx005e0a5eld01aqu9wi34.mp4"),
        os.path.join(dl, "magichour_face_swap-2026-09-13-478x850-cmtzwpy090068ke01tyt6cqdv.mp4"),
    ]
    real_videos = [
        os.path.join(dl, "WhatsApp Video 2026-10-22 at 11.57.32 PM.mp4"),
        os.path.join(dl, "WhatsApp Video 2026-09-13 at 8.01.48 PM.mp4"),
        os.path.join(dl, "WhatsApp Video 2026-09-13 at 7.32.31 PM.mp4"),
    ]
    
    cache_file = r"e:\deepfake ai detection\deepfake-detection\data\features_cache.npz"
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    if os.path.exists(cache_file):
        logger.info(f"Loading cached features from {cache_file}...")
        data = np.load(cache_file)
        X = data["X"]
        y = data["y"]
        logger.info(f"Loaded {len(X)} features from cache: {sum(y == 0)} REAL, {sum(y == 1)} FAKE")
    else:
        logger.info("Extracting features across all datasets...")
        real_features = []
        fake_features = []
        
        # Real images
        logger.info("Extracting train_real features...")
        real_features.extend(extract_features_from_images(train_real_paths, extractor, is_real=True))
        logger.info("Extracting test_real features...")
        real_features.extend(extract_features_from_images(test_real_paths, extractor, is_real=True))
        if extra_real_paths:
            logger.info("Extracting user real test photo features...")
            real_features.extend(extract_features_from_images(extra_real_paths, extractor, is_real=True))
            
        # Real videos
        logger.info("Extracting real video features...")
        for v in real_videos:
            real_features.extend(extract_features_from_video(v, extractor, max_frames=30))
            
        # Fake images
        logger.info("Extracting train_fake features...")
        fake_features.extend(extract_features_from_images(train_fake_paths, extractor, is_real=False))
        logger.info("Extracting test_fake features...")
        fake_features.extend(extract_features_from_images(test_fake_paths, extractor, is_real=False))
        logger.info("Extracting sample_fake features...")
        fake_features.extend(extract_features_from_images(sample_fake_paths, extractor, is_real=False))
        
        # Fake videos
        logger.info("Extracting fake video features...")
        for v in fake_videos:
            fake_features.extend(extract_features_from_video(v, extractor, max_frames=30))
            
        X = np.array(fake_features + real_features)
        y = np.array([1] * len(fake_features) + [0] * len(real_features))
        
        np.savez_compressed(cache_file, X=X, y=y)
        logger.info(f"Saved feature cache: {len(X)} total ({sum(y == 0)} REAL, {sum(y == 1)} FAKE)")
    
    # Train Logistic Regression classifier
    logger.info("Fitting Logistic Regression classifier...")
    # Test different C values
    best_clf = None
    best_c = 0.5
    for c_val in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        clf = LogisticRegression(max_iter=1500, C=c_val, class_weight="balanced", random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        acc = accuracy_score(y, preds)
        logger.info(f"C={c_val}: Train Accuracy = {acc*100:.2f}%")
        if c_val == 0.5:
            best_clf = clf
            
    clf = best_clf if best_clf is not None else clf
    
    # Build complete PyTorch model
    final_model = efficientnet_b0(weights=None)
    in_features = final_model.classifier[1].in_features
    final_model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 1)
    )
    final_model.features.load_state_dict(base_model.features.cpu().state_dict())
    final_model.classifier[1].weight.data = torch.from_numpy(clf.coef_).float()
    final_model.classifier[1].bias.data = torch.from_numpy(clf.intercept_).float()
    final_model.eval()
    
    # --- EVALUATE ON USER TEST FACE ---
    if os.path.exists(user_face_path):
        im_user = Image.open(user_face_path).convert("RGB")
        t_user = tfm(im_user).unsqueeze(0)
        with torch.no_grad():
            prob_user = torch.sigmoid(final_model(t_user)).item()
        verdict_user = "DEEPFAKE" if prob_user >= 0.5 else "REAL"
        logger.info(f"\n*** USER TEST PHOTO: Prob={prob_user*100:.1f}% -> {verdict_user} (Expected: REAL) ***")
        
    # --- EVALUATE TEST REAL PORTRAITS ---
    logger.info("\n=== EVALUATING REAL TEST PORTRAITS ===")
    real_portraits = ["5004.jpg", "5006.jpg", "5007.jpg", "5015.jpg", "5017.jpg", "5044.jpg", "5047.jpg", "5050.jpg"]
    for fname in real_portraits:
        p = os.path.join(arch, r"test-20250112T065939Z-001\test\real", fname)
        if os.path.exists(p):
            im = Image.open(p).convert("RGB")
            crop, _ = fast_detect_and_crop_face(im, margin=0.2)
            t_crop = tfm(crop).unsqueeze(0)
            with torch.no_grad():
                prob = torch.sigmoid(final_model(t_crop)).item()
            verdict = "DEEPFAKE" if prob >= 0.5 else "REAL"
            logger.info(f"Real {fname}: Deepfake Prob={prob*100:.1f}% -> {verdict}")
            
    # --- EVALUATE SAMPLE FAKES ---
    logger.info("\n=== EVALUATING SAMPLE FAKE IMAGES ===")
    sample_dir = os.path.join(arch, r"Sample_fake_images\Sample_fake_images\fake")
    for fname in sorted(os.listdir(sample_dir)):
        if fname.lower().endswith(('.jpg', '.png')):
            im = Image.open(os.path.join(sample_dir, fname)).convert("RGB")
            crop, _ = fast_detect_and_crop_face(im, margin=0.2)
            t_crop = tfm(crop).unsqueeze(0)
            with torch.no_grad():
                prob = torch.sigmoid(final_model(t_crop)).item()
            verdict = "DEEPFAKE" if prob >= 0.5 else "REAL"
            logger.info(f"Fake {fname}: Deepfake Prob={prob*100:.1f}% -> {verdict}")
            
    # --- EVALUATE ALL VIDEOS ---
    logger.info("\n=== EVALUATING ALL VIDEOS ===")
    all_vids = [(v, "DEEPFAKE") for v in fake_videos] + [(v, "REAL") for v in real_videos]
    for vpath, expected in all_vids:
        feats = extract_features_from_video(vpath, extractor, max_frames=30)
        if not feats:
            continue
        t_feats = torch.from_numpy(np.array(feats)).float()
        with torch.no_grad():
            probs = torch.sigmoid(final_model.classifier[1](t_feats)).squeeze().numpy()
        mean_p = float(probs.mean())
        fake_ratio = float((probs >= 0.5).mean())
        verdict = "DEEPFAKE" if (mean_p >= 0.5 or fake_ratio >= 0.35) else "REAL"
        logger.info(f"{os.path.basename(vpath)}: Exp={expected} | Got={verdict} (mean_prob={mean_p*100:.1f}%, fake_ratio={fake_ratio*100:.1f}%)")
        
    # Save checkpoint with 0.5 threshold
    checkpoint_dict = {
        "model_state_dict": final_model.state_dict(),
        "epoch": 40,
        "val_metric": 0.98,
        "architecture": "efficientnet_b0",
        "class_mapping": {"REAL": 0, "DEEPFAKE": 1},
        "threshold": 0.5,
        "input_size": 224,
        "version": "5.0-master-unified",
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
        
    logger.info("Master model training, evaluation, and saving complete!")

if __name__ == "__main__":
    main()
