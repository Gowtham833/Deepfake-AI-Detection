#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse
import logging
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from src.models.efficientnet import create_efficientnet_b0
from src.training.evaluate import Evaluator
from src.training.threshold import find_optimal_threshold
from src.data.dataset import DeepfakeDataset
from src.utils.io import load_config, load_json, ensure_dir
from src.utils.device import get_device

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    ensure_dir(args.output_dir)

    manifest = load_json(args.manifest)
    dataset = DeepfakeDataset(manifest[args.split], is_training=False)
    dataloader = DataLoader(dataset, batch_size=config.get("training", {}).get("batch_size", 32), shuffle=False)

    model = create_efficientnet_b0(pretrained=False, num_classes=2).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    
    evaluator = Evaluator(model, dataloader, device)
    metrics, y_true, y_prob = evaluator.evaluate()
    
    if args.threshold is None:
        logger.info("Finding optimal threshold...")
        val_dataset = DeepfakeDataset(manifest["val"], is_training=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        val_evaluator = Evaluator(model, val_loader, device)
        _, val_y_true, val_y_prob = val_evaluator.evaluate()
        threshold = find_optimal_threshold(val_y_true, val_y_prob)
        logger.info(f"Optimal threshold found: {threshold:.4f}")
    else:
        threshold = args.threshold
        
    y_pred = (y_prob >= threshold).astype(int)
    
    report = {
        "metrics": metrics,
        "threshold": float(threshold)
    }
    with open(Path(args.output_dir) / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    logger.info("Evaluation complete. Report saved.")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
