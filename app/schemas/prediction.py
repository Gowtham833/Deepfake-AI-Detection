from typing import List, Optional
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    """Response model for prediction requests."""
    model_config = {"protected_namespaces": ()}
    prediction: str
    real_probability: float
    deepfake_probability: float
    threshold: float
    model_name: str
    face_detected: bool
    face_count: int
    selected_face_box: Optional[List[int]] = None
    explanation_available: bool = False
    processed_face_image: Optional[str] = None
    gradcam_heatmap: Optional[str] = None
    gradcam_overlay: Optional[str] = None
    warnings: List[str] = []


class HealthResponse(BaseModel):
    """Response model for health check."""
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    device: str
    message: str


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str
    detail: Optional[str] = None


class FrameResult(BaseModel):
    """Result for a single video frame."""
    frame_index: int
    timestamp_seconds: float = 0.0
    face_detected: bool
    face_count: int = 0
    deepfake_probability: Optional[float] = None
    real_probability: Optional[float] = None
    prediction: Optional[str] = None


class VideoPredictionResponse(BaseModel):
    """Response model for video prediction requests."""
    model_config = {"protected_namespaces": ()}
    prediction: str
    deepfake_probability: float
    real_probability: float
    threshold: float
    model_name: str
    total_frames_extracted: int
    frames_with_faces: int
    frames_analyzed: int
    fake_frame_count: int = 0
    fake_frame_ratio: float = 0.0
    frame_results: List[FrameResult] = []
    video_duration_seconds: float = 0.0
    video_fps: float = 0.0
    video_resolution: List[int] = []
    warnings: List[str] = []
