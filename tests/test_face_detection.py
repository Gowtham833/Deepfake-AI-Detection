import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from src.data.face_detection import FaceDetector, detect_faces, select_face, crop_face

@patch('src.data.face_detection.MTCNN')
def test_valid_image(mock_mtcnn):
    detector = FaceDetector()
    detector.mtcnn = mock_mtcnn.return_value
    detector.mtcnn.detect.return_value = (np.array([[10, 10, 50, 50]]), np.array([0.99]))
    
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = detect_faces(img, detector)
    
    assert detections is not None
    assert len(detections) == 1

@patch('src.data.face_detection.MTCNN')
def test_no_face_case(mock_mtcnn):
    detector = FaceDetector()
    detector.mtcnn = mock_mtcnn.return_value
    detector.mtcnn.detect.return_value = (None, None)
    
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = detect_faces(img, detector)
    
    assert len(detections) == 0

@patch('src.data.face_detection.MTCNN')
def test_multi_face_case(mock_mtcnn):
    detector = FaceDetector()
    detector.mtcnn = mock_mtcnn.return_value
    detector.mtcnn.detect.return_value = (np.array([[10, 10, 30, 30], [50, 50, 90, 90]]), np.array([0.9, 0.95]))
    
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = detect_faces(img, detector)
    
    assert len(detections) == 2

@patch('src.data.face_detection.MTCNN')
def test_largest_face_selection(mock_mtcnn):
    detector = FaceDetector()
    detector.mtcnn = mock_mtcnn.return_value
    # Box 1: 20x20 = 400 area, Box 2: 40x40 = 1600 area
    detector.mtcnn.detect.return_value = (np.array([[10, 10, 30, 30], [50, 50, 90, 90]]), np.array([0.9, 0.95]))
    
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = detect_faces(img, detector)
    selected = select_face(detections, strategy='largest')
    
    assert selected is not None
    assert selected.box == [50, 50, 90, 90]
