#!/usr/bin/env python3
"""
Quick inference script to test model on a directory of images.

Usage:
    python scripts/run_inference.py --image-dir "C:/path/to/images" 
    python scripts/run_inference.py --image "C:/path/to/single/image.jpg"
    
    # Test with sample fake images:
    python scripts/run_inference.py --image-dir "C:/Users/Dell/Downloads/archive/Sample_fake_images/Sample_fake_images/fake"
    
    # Test with test set:
    python scripts/run_inference.py --image-dir "C:/Users/Dell/Downloads/archive/test-20250112T065939Z-001/test/fake" --label fake
    python scripts/run_inference.py --image-dir "C:/Users/Dell/Downloads/archive/test-20250112T065939Z-001/test/real" --label real
"""
import sys
import os
from pathlib import Path
import argparse
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from src.models.efficientnet import create_model
from src.utils.device import get_device

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


def load_model(checkpoint_path: str, device):
    config = {"model": {"pretrained": False, "dropout_rate": 0.4}}
    model = create_model(config).to(device)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        val_metric = ckpt.get('val_metric') or ckpt.get('val_loss')
        metric_str = f"{val_metric:.4f}" if isinstance(val_metric, (int, float)) else "N/A"
        logger.info(f"Loaded checkpoint: epoch={ckpt.get('epoch', 'N/A')}, "
                    f"val_metric={metric_str}")
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    return model


def get_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict_image(model, image_path: str, transform, device, threshold: float = 0.4):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        return None, None, str(e)
    
    tensor = transform(img).unsqueeze(0).to(device)
    logit = model(tensor).squeeze()
    prob_fake = torch.sigmoid(logit).item()
    prob_real = 1.0 - prob_fake
    
    label = "DEEPFAKE" if prob_fake >= threshold else "REAL"
    return label, {"real": prob_real, "fake": prob_fake}, None


def main():
    parser = argparse.ArgumentParser(description="Run deepfake inference on images")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Single image to analyze")
    group.add_argument("--image-dir", type=str, help="Directory of images to analyze")
    group.add_argument("--video", type=str, help="Video file to analyze")
    parser.add_argument("--checkpoint", type=str,
                        default=r"e:\deepfake ai detection\deepfake-detection\models\best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--label", type=str, choices=["real", "fake"],
                        help="Ground truth label for directory (for metrics)")
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Device: {device}")

    model   = load_model(args.checkpoint, device)
    tfm     = get_transform(args.image_size)

    if args.video:
        from app.services.video_inference_service import VideoInferenceService
        video_service = VideoInferenceService()
        if not video_service.is_ready:
            print("ERROR: Model failed to load.")
            return
        
        with open(args.video, "rb") as f:
            v_bytes = f.read()
        
        print(f"\nAnalyzing Video: {args.video} ({len(v_bytes) / (1024*1024):.1f} MB)")
        result = video_service.process_video(v_bytes)
        
        print("-" * 70)
        print(f"Overall Verdict: {result['prediction']}")
        print(f"Deepfake Prob  : {result['deepfake_probability']*100:.1f}%")
        print(f"Real Prob      : {result['real_probability']*100:.1f}%")
        print(f"Frames Analyzed: {result['frames_analyzed']} / {result['total_frames_extracted']}")
        print(f"Fake Frames    : {result.get('fake_frame_count', 0)} ({result.get('fake_frame_ratio', 0)*100:.1f}%)")
        print(f"Duration       : {result['video_duration_seconds']:.1f}s, FPS: {result['video_fps']}")
        
        if result.get("frame_results"):
            print("-" * 70)
            print(f"{'Frame':<8} {'Time':<8} {'Face':<8} {'Result':<12} {'Fake%':>8}")
            print("-" * 70)
            for fr in result["frame_results"]:
                t_str = f"{fr['timestamp_seconds']:.1f}s"
                face_str = f"Yes ({fr['face_count']})" if fr["face_detected"] else "No"
                pred_str = fr.get("prediction") or "N/A"
                fake_str = f"{fr['deepfake_probability']*100:.1f}%" if fr.get("deepfake_probability") is not None else "N/A"
                print(f"#{fr['frame_index']+1:<7} {t_str:<8} {face_str:<8} {pred_str:<12} {fake_str:>8}")
        
        if result.get("warnings"):
            print("\nWarnings:")
            for w in result["warnings"]:
                print(f"  - {w}")
        return

    if args.image:
        label, probs, err = predict_image(model, args.image, tfm, device, args.threshold)
        if err:
            print(f"ERROR: {err}")
        else:
            print(f"\nFile    : {args.image}")
            print(f"Result  : {label}")
            print(f"Fake %  : {probs['fake']*100:.1f}%")
            print(f"Real %  : {probs['real']*100:.1f}%")
        return

    # Directory mode
    image_dir = Path(args.image_dir)
    image_files = [f for f in image_dir.iterdir()
                   if f.suffix.lower() in EXTENSIONS and f.is_file()]

    if not image_files:
        logger.error(f"No images found in {image_dir}")
        sys.exit(1)

    print(f"\nAnalyzing {len(image_files)} images from: {image_dir}")
    print("-" * 70)
    print(f"{'File':<40} {'Result':<12} {'Fake%':>8}  {'Real%':>8}")
    print("-" * 70)

    all_probs, all_preds = [], []
    errors = 0

    for img_path in sorted(image_files):
        label, probs, err = predict_image(model, str(img_path), tfm, device, args.threshold)
        if err:
            print(f"{img_path.name:<40} ERROR: {err}")
            errors += 1
            continue

        all_probs.append(probs['fake'])
        all_preds.append(1 if label == "DEEPFAKE" else 0)

        # Color indicator
        indicator = "🔴 FAKE" if label == "DEEPFAKE" else "🟢 REAL"
        print(f"{img_path.name:<40} {indicator:<12} {probs['fake']*100:>7.1f}%  {probs['real']*100:>7.1f}%")

    print("-" * 70)
    n_fake = sum(all_preds)
    n_real = len(all_preds) - n_fake
    print(f"\nSummary: {n_fake} DEEPFAKE, {n_real} REAL  ({errors} errors)")

    if args.label and all_preds:
        true_label = 1 if args.label == "fake" else 0
        true_labels = [true_label] * len(all_preds)
        print("\nClassification Report:")
        print(classification_report(true_labels, all_preds,
                                    target_names=['REAL', 'FAKE'], zero_division=0))
        try:
            auc = roc_auc_score(true_labels, all_probs)
            print(f"AUC: {auc:.4f}")
        except Exception:
            pass

        cm = confusion_matrix(true_labels, all_preds)
        print(f"\nConfusion Matrix:\n{cm}")


if __name__ == "__main__":
    main()
