import cv2
import numpy as np
import logging
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights
from .face_detection import FaceDetector, detect_faces, select_face, crop_face

logger = logging.getLogger(__name__)

def get_inference_transform(config):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    return weights.transforms()

def get_training_transform(config):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    # Base transforms from weights, with some augmentations
    base_transforms = weights.transforms()
    
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        base_transforms
    ])
    return train_transforms

def load_and_validate_image(path):
    # Load using cv2, convert to RGB
    img = cv2.imread(path)
    if img is None:
        logger.warning(f"Could not read image {path}")
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def preprocess_image(image_rgb, config, detector=None):
    if detector is None:
        detector = FaceDetector()
        
    detections = detect_faces(image_rgb, detector)
    face_count = len(detections)
    
    if face_count == 0:
        return None, {
            'face_detected': False,
            'face_count': 0,
            'selected_face_box': None,
            'preprocess_status': 'no_face'
        }
        
    selected = select_face(detections, strategy=config.get('face_selection_strategy', 'largest'))
    cropped = crop_face(image_rgb, selected.box, margin=config.get('face_crop_margin', 0.2))
    
    # Convert numpy array to PIL for torchvision transforms
    from PIL import Image
    pil_img = Image.fromarray(cropped)
    
    transform = get_inference_transform(config)
    tensor = transform(pil_img)
    
    return tensor, {
        'face_detected': True,
        'face_count': face_count,
        'selected_face_box': selected.box,
        'confidence': selected.confidence,
        'preprocess_status': 'success'
    }
