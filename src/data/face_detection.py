import cv2
import numpy as np
from facenet_pytorch import MTCNN
from collections import namedtuple
import logging
import torch
from src.utils.device import get_device

logger = logging.getLogger(__name__)

FaceDetection = namedtuple('FaceDetection', ['box', 'confidence'])

class FaceDetector:
    def __init__(self, device=None, min_face_size=40, thresholds=[0.6, 0.7, 0.7]):
        if device is None:
            device = get_device()
        
        # MTCNN with min_face_size filter to avoid false-positive background textures
        self.mtcnn = MTCNN(
            keep_all=True,
            min_face_size=min_face_size,
            thresholds=thresholds,
            device=device
        )

def detect_faces(image, detector=None):
    if detector is None:
        detector = FaceDetector().mtcnn
    elif hasattr(detector, 'mtcnn'):
        detector = detector.mtcnn
        
    # Ensure image is RGB PIL Image or numpy array
    try:
        boxes, probs = detector.detect(image)
        detections = []
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob is not None and prob >= 0.85:
                    x1, y1, x2, y2 = box
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    # Filter out degenerate boxes
                    if w >= 10 and h >= 10 and 0.2 <= (w / max(h, 1)) <= 5.0:
                        detections.append(FaceDetection(box.tolist(), float(prob)))
        return detections
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return []

def select_face(detections, strategy='largest'):
    if not detections:
        return None
        
    if strategy == 'largest':
        def area(box):
            return max(0, box[2] - box[0]) * max(0, box[3] - box[1])
        return max(detections, key=lambda d: area(d.box))
    elif strategy == 'highest_confidence':
        return max(detections, key=lambda d: d.confidence)
        
    return detections[0]

def crop_face(image, box, margin=0.2):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    
    margin_x = w * margin
    margin_y = h * margin
    
    x1 = max(0, int(x1 - margin_x))
    y1 = max(0, int(y1 - margin_y))
    x2 = min(image.shape[1], int(x2 + margin_x))
    y2 = min(image.shape[0], int(y2 + margin_y))
    
    return image[y1:y2, x1:x2]
