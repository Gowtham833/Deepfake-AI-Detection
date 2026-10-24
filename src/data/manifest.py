import os
import pandas as pd
import logging
from PIL import Image

logger = logging.getLogger(__name__)

REAL = 0
DEEPFAKE = 1
class_names = ["REAL", "DEEPFAKE"]

def validate_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

def generate_image_manifest(data_root, real_dir, fake_dir, config):
    records = []
    
    for cls_dir, label, cname in [(real_dir, REAL, "REAL"), (fake_dir, DEEPFAKE, "DEEPFAKE")]:
        full_dir = os.path.join(data_root, cls_dir)
        if not os.path.exists(full_dir):
            logger.warning(f"Directory {full_dir} does not exist.")
            continue
            
        for root, _, files in os.walk(full_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(root, file)
                    if validate_image(path):
                        records.append({
                            'image_path': path,
                            'label': label,
                            'class_name': cname,
                            'source_video': None,
                            'split': None,
                            'width': None,
                            'height': None,
                            'face_detected': None,
                            'face_count': None,
                            'selected_face_box': None,
                            'preprocess_status': None,
                            'failure_reason': None
                        })
                    else:
                        logger.warning(f"Invalid image: {path}")

    return pd.DataFrame(records)

def generate_video_manifest(video_real_dir, video_fake_dir, frames_dir, config):
    # Generates manifest from extracted frames
    records = []
    
    for cls_dir, label, cname in [(video_real_dir, REAL, "REAL"), (video_fake_dir, DEEPFAKE, "DEEPFAKE")]:
        full_dir = os.path.join(frames_dir, os.path.basename(cls_dir))
        if not os.path.exists(full_dir):
            logger.warning(f"Directory {full_dir} does not exist.")
            continue
            
        for root, _, files in os.walk(full_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(root, file)
                    if validate_image(path):
                        # Assuming source video is part of the filename or directory structure
                        # e.g. video_name_frame_001.jpg -> video_name
                        source_video = file.rsplit('_frame_', 1)[0]
                        records.append({
                            'image_path': path,
                            'label': label,
                            'class_name': cname,
                            'source_video': source_video,
                            'split': None,
                            'width': None,
                            'height': None,
                            'face_detected': None,
                            'face_count': None,
                            'selected_face_box': None,
                            'preprocess_status': None,
                            'failure_reason': None
                        })
                    else:
                        logger.warning(f"Invalid image: {path}")

    return pd.DataFrame(records)

def load_manifest(path):
    return pd.read_csv(path)

def save_manifest(df, path):
    df.to_csv(path, index=False)
    logger.info(f"Manifest saved to {path}")
