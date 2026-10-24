import os
import io
import cv2
import torch
import numpy as np
import tempfile
import logging
from PIL import Image
from typing import List, Dict, Any, Optional

from app.services.inference_service import InferenceService
from src.data.face_detection import detect_faces, select_face, crop_face

logger = logging.getLogger(__name__)


class VideoInferenceService:
    """Service for deepfake detection on video files."""
    
    def __init__(self):
        self.inference_service = InferenceService()
    
    @property
    def is_ready(self) -> bool:
        return self.inference_service.is_model_loaded
    
    def process_video(self, video_bytes: bytes, max_frames: int = 30, extraction_fps: float = 1.0) -> dict:
        """
        Process a video for deepfake detection.
        
        Args:
            video_bytes: Raw video file bytes
            max_frames: Maximum number of frames to analyze
            extraction_fps: Frames per second to extract
            
        Returns:
            dict with keys:
                - prediction: "DEEPFAKE" or "REAL"
                - deepfake_probability: float (mean across frames)
                - real_probability: float
                - threshold: float
                - model_name: str
                - total_frames_extracted: int
                - frames_with_faces: int
                - frames_analyzed: int
                - frame_results: list of per-frame dicts (frame_index, deepfake_probability, real_probability, prediction, face_detected)
                - video_duration_seconds: float
                - video_fps: float
                - video_resolution: [width, height]
                - warnings: list of strings
        """
        if not self.is_ready:
            raise RuntimeError("Model is not loaded")
        
        warnings = []
        
        # Write video bytes to a temp file so OpenCV can read it
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        
        cap = None
        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file. Ensure it is a valid video format.")
            
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_video_frames / video_fps if video_fps > 0 else 0
            
            # Calculate frame interval
            frame_interval = max(1, int(video_fps / extraction_fps))
            
            frame_results = []
            frame_idx = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    # Process this frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)
                    
                    # Detect faces
                    detections = detect_faces(pil_frame, self.inference_service.face_detector)
                    
                    frame_result = {
                        "frame_index": extracted_count,
                        "timestamp_seconds": round(frame_idx / video_fps, 2) if video_fps > 0 else 0,
                        "face_detected": len(detections) > 0,
                        "face_count": len(detections),
                    }
                    
                    if len(detections) > 0:
                        selected = select_face(detections, strategy="largest")
                        face_np = crop_face(frame_rgb, selected.box, margin=0.2)
                        face_pil = Image.fromarray(face_np)
                        
                        # Run inference on cropped face
                        input_tensor = self.inference_service.transform(face_pil).unsqueeze(0).to(self.inference_service.device)
                        with torch.no_grad():
                            outputs = self.inference_service.model(input_tensor)
                            prob_deepfake = torch.sigmoid(outputs).item()
                        
                        frame_result["deepfake_probability"] = round(prob_deepfake, 4)
                        frame_result["real_probability"] = round(1.0 - prob_deepfake, 4)
                        frame_result["prediction"] = "DEEPFAKE" if prob_deepfake >= self.inference_service.threshold else "REAL"
                    else:
                        # Pre-compute fallback prediction on full frame in case no faces are detected in video
                        input_tensor = self.inference_service.transform(pil_frame).unsqueeze(0).to(self.inference_service.device)
                        with torch.no_grad():
                            outputs = self.inference_service.model(input_tensor)
                            prob_fallback = torch.sigmoid(outputs).item()
                        
                        frame_result["fallback_probability"] = round(prob_fallback, 4)
                        frame_result["deepfake_probability"] = None
                        frame_result["real_probability"] = None
                        frame_result["prediction"] = None
                    
                    frame_results.append(frame_result)
                    extracted_count += 1
                    
                    if extracted_count >= max_frames:
                        break
                
                frame_idx += 1
            
            cap.release()
            
            # Aggregate results
            frames_with_faces = [f for f in frame_results if f["face_detected"]]
            
            if not frames_with_faces:
                # Fallback: if no faces detected in any frame, use full-frame analysis
                warnings.append("No clear face detected by MTCNN in video frames — analyzed full frames as fallback.")
                for f in frame_results:
                    fb = f.get("fallback_probability", 0.0)
                    f["deepfake_probability"] = fb
                    f["real_probability"] = round(1.0 - fb, 4)
                    f["prediction"] = "DEEPFAKE" if fb >= self.inference_service.threshold else "REAL"
                
                fake_probs = [f["deepfake_probability"] for f in frame_results]
                mean_fake_prob = sum(fake_probs) / len(fake_probs) if fake_probs else 0.0
                fake_frame_count = sum(1 for f in frame_results if f["prediction"] == "DEEPFAKE")
                fake_frame_ratio = fake_frame_count / len(frame_results) if frame_results else 0.0
                overall_prediction = "DEEPFAKE" if mean_fake_prob >= self.inference_service.threshold else "REAL"
                
                return {
                    "prediction": overall_prediction,
                    "deepfake_probability": round(mean_fake_prob, 4),
                    "real_probability": round(1.0 - mean_fake_prob, 4),
                    "threshold": self.inference_service.threshold,
                    "model_name": "EfficientNet-B0",
                    "total_frames_extracted": extracted_count,
                    "frames_with_faces": 0,
                    "frames_analyzed": len(frame_results),
                    "fake_frame_count": fake_frame_count,
                    "fake_frame_ratio": round(fake_frame_ratio, 4),
                    "frame_results": frame_results,
                    "video_duration_seconds": round(duration, 2),
                    "video_fps": round(video_fps, 2),
                    "video_resolution": [width, height],
                    "warnings": warnings,
                }
            
            # Mean aggregation
            fake_probs = [f["deepfake_probability"] for f in frames_with_faces]
            mean_fake_prob = sum(fake_probs) / len(fake_probs) if fake_probs else 0.0
            
            # Count how many individual frames were classified as DEEPFAKE
            fake_frame_count = sum(1 for f in frames_with_faces if f["prediction"] == "DEEPFAKE")
            fake_frame_ratio = fake_frame_count / len(frames_with_faces) if frames_with_faces else 0.0
            
            # Robust video deepfake verdict:
            # Video is flagged as DEEPFAKE if mean probability exceeds threshold OR at least 35% of faces show manipulation
            is_deepfake = (mean_fake_prob >= self.inference_service.threshold) or (fake_frame_ratio >= 0.35)
            overall_prediction = "DEEPFAKE" if is_deepfake else "REAL"
            
            # Display probability consistent with the verdict
            if is_deepfake:
                display_deepfake_prob = max(mean_fake_prob, fake_frame_ratio)
            else:
                display_deepfake_prob = mean_fake_prob
            
            if len(frames_with_faces) < extracted_count:
                warnings.append(f"{extracted_count - len(frames_with_faces)} frame(s) had no detectable face and were excluded from analysis.")
            
            return {
                "prediction": overall_prediction,
                "deepfake_probability": round(display_deepfake_prob, 4),
                "real_probability": round(1.0 - display_deepfake_prob, 4),
                "threshold": self.inference_service.threshold,
                "model_name": "EfficientNet-B0",
                "total_frames_extracted": extracted_count,
                "frames_with_faces": len(frames_with_faces),
                "frames_analyzed": len(frames_with_faces),
                "fake_frame_count": fake_frame_count,
                "fake_frame_ratio": round(fake_frame_ratio, 4),
                "frame_results": frame_results,
                "video_duration_seconds": round(duration, 2),
                "video_fps": round(video_fps, 2),
                "video_resolution": [width, height],
                "warnings": warnings,
            }
        
        finally:
            # Clean up OpenCV capture and temp file
            if cap is not None:
                cap.release()
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
