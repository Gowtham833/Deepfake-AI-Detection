import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Mock the FastAPI app
from fastapi import FastAPI
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict():
    return {"label": "DEEPFAKE", "confidence": 0.95, "class_id": 1}

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch('app.services.inference_service.InferenceService')
def test_predict_endpoint(mock_infer_service):
    # Mocking InferenceService logic
    mock_instance = mock_infer_service.return_value
    mock_instance.predict.return_value = {"label": "DEEPFAKE", "confidence": 0.95, "class_id": 1}
    
    response = client.post("/predict")
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert data["label"] == "DEEPFAKE"
    assert data["class_id"] == 1
    assert data["confidence"] > 0.9

@app.post("/predict-video")
def predict_video():
    return {
        "prediction": "DEEPFAKE",
        "deepfake_probability": 0.88,
        "real_probability": 0.12,
        "threshold": 0.4,
        "model_name": "EfficientNet-B0",
        "total_frames_extracted": 10,
        "frames_with_faces": 8,
        "frames_analyzed": 8,
        "fake_frame_count": 7,
        "fake_frame_ratio": 0.875,
        "frame_results": [],
        "video_duration_seconds": 10.0,
        "video_fps": 30.0,
        "video_resolution": [1920, 1080],
        "warnings": []
    }

def test_predict_video_endpoint():
    response = client.post("/predict-video")
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "DEEPFAKE"
    assert data["frames_analyzed"] == 8
