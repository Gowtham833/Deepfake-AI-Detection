"""
Data pipeline components for the deepfake detection project.
"""

from .dataset import DeepfakeDataset, custom_collate_fn, create_dataloaders
from .manifest import generate_image_manifest, generate_video_manifest, load_manifest, save_manifest, validate_image
from .extract_frames import extract_frames_from_video, extract_frames_from_directory
from .face_detection import FaceDetector, FaceDetection, detect_faces, select_face, crop_face
from .preprocessing import get_inference_transform, get_training_transform, preprocess_image, load_and_validate_image
from .split import split_dataset, group_aware_split, simple_split
