#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse
import logging
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.manifest import generate_manifest, update_manifest_with_faces
from src.data.extract_frames import process_videos_in_directory
from src.data.face_detection import FaceDetector
from src.data.split import split_dataset_group_aware
from src.utils.io import load_config, ensure_dir, save_json
from src.utils.seed import set_seed
from src.utils.device import get_device

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for deepfake detection")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory of the raw data")
    parser.add_argument("--mode", type=str, choices=["image", "video"], required=True, help="Mode of data (image or video)")
    parser.add_argument("--output-manifest", type=str, required=True, help="Path to save the final manifest")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    data_root = Path(args.data_root)
    if not data_root.exists():
        logger.error(f"Data root {data_root} does not exist.")
        sys.exit(1)

    if args.mode == "video":
        logger.info("Extracting frames from videos...")
        frames_dir = data_root.parent / f"{data_root.name}_frames"
        process_videos_in_directory(
            input_dir=data_root,
            output_dir=frames_dir,
            frames_per_video=config.get("data", {}).get("frames_per_video", 10)
        )
        data_to_scan = frames_dir
    else:
        data_to_scan = data_root

    logger.info("Generating manifest...")
    manifest = generate_manifest(data_to_scan)
    
    logger.info("Running face detection...")
    device = get_device()
    detector = FaceDetector(device=device)
    manifest = update_manifest_with_faces(manifest, detector)
    
    logger.info("Splitting dataset...")
    train_manifest, val_manifest, test_manifest = split_dataset_group_aware(
        manifest, 
        train_ratio=config.get("data", {}).get("train_ratio", 0.7),
        val_ratio=config.get("data", {}).get("val_ratio", 0.15),
        test_ratio=config.get("data", {}).get("test_ratio", 0.15)
    )
    
    final_manifest = {
        "train": train_manifest,
        "val": val_manifest,
        "test": test_manifest
    }
    
    output_path = Path(args.output_manifest)
    ensure_dir(output_path.parent)
    save_json(final_manifest, output_path)
    
    logger.info("Dataset preparation complete.")
    logger.info(f"Train samples: {len(train_manifest)}")
    logger.info(f"Val samples: {len(val_manifest)}")
    logger.info(f"Test samples: {len(test_manifest)}")

if __name__ == "__main__":
    main()
