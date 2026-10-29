import os
import io
import gc
import cv2
import torch
import numpy as np
import tempfile
import logging
from PIL import Image
from typing import List, Dict, Any, Optional, Union

from app.services.inference_service import InferenceService
from src.data.face_detection import detect_faces, select_face, crop_face

logger = logging.getLogger(__name__)


class VideoInferenceService:
    """Service for deepfake detection on video files with high-speed seeking and memory optimization."""
    
    def __init__(self):
        self.inference_service = InferenceService()
    
    @property
    def is_ready(self) -> bool:
        return self.inference_service.is_model_loaded
    
    def process_video(
        self, 
        video_input: Union[bytes, str], 
        max_frames: int = 8, 
        extraction_fps: float = 1.0
    ) -> dict:
        """
        Process a video for deepfake detection with direct seeking and downscaled face detection.
        
        Args:
            video_input: Raw video bytes or a path to a video file
            max_frames: Maximum number of frames to analyze (default 8: optimal balance between accuracy & speed on cloud CPU)
            extraction_fps: Frames per second reference
            
        Returns:
            dict containing overall verdict, probabilities, frame-by-frame stats, and duration.
        """
        if not self.is_ready:
            raise RuntimeError("Model is not loaded")
        
        warnings = []
        created_temp_file = False
        
        if isinstance(video_input, str):
            tmp_path = video_input
        else:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp.write(video_input)
                tmp_path = tmp.name
                created_temp_file = True
        
        cap = None
        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file. Ensure it is a valid video format.")
            
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = round(total_video_frames / video_fps, 2) if (video_fps > 0 and total_video_frames > 0) else 0.0
            
            # Determine target frame indices using direct seek
            num_samples = max(1, min(max_frames, total_video_frames if total_video_frames > 0 else max_frames))
            
            if total_video_frames > 1:
                # Evenly distributed across 5% to 95% of video duration to capture faces in natural flow
                sample_indices = np.linspace(
                    int(total_video_frames * 0.05), 
                    int(total_video_frames * 0.95), 
                    num=num_samples, 
                    dtype=int
                ).tolist()
                sample_indices = sorted(list(set(sample_indices)))
            elif total_video_frames == 1:
                sample_indices = [0]
            else:
                sample_indices = []
            
            frame_results = []
            extracted_count = 0
            
            # FAST PATH: Direct frame seeking (bypasses decoding hundreds of unnecessary intermediate frames)
            if sample_indices:
                for idx in sample_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue
                    
                    res = self._analyze_single_frame(frame, extracted_count, idx, video_fps)
                    frame_results.append(res)
                    extracted_count += 1
                    del frame
            else:
                # FALLBACK PATH: Sequential read if frame count could not be determined from video header
                step = max(1, int(video_fps))
                curr_idx = 0
                while extracted_count < max_frames:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    if curr_idx % step == 0:
                        res = self._analyze_single_frame(frame, extracted_count, curr_idx, video_fps)
                        frame_results.append(res)
                        extracted_count += 1
                    curr_idx += 1
                    del frame
            
            cap.release()
            cap = None
            gc.collect()
            
            if not frame_results:
                raise ValueError("Could not extract any valid video frames. The video stream may be corrupted or unreadable.")
            
            # Aggregate results
            frames_with_faces = [f for f in frame_results if f["face_detected"]]
            
            if not frames_with_faces:
                warnings.append("No clear face detected in sampled video frames. Analyzed full frames as fallback.")
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
                    "video_duration_seconds": duration,
                    "video_fps": round(video_fps, 2),
                    "video_resolution": [width, height],
                    "warnings": warnings,
                }
            
            fake_probs = [f["deepfake_probability"] for f in frames_with_faces]
            mean_fake_prob = sum(fake_probs) / len(fake_probs) if fake_probs else 0.0
            
            fake_frame_count = sum(1 for f in frames_with_faces if f["prediction"] == "DEEPFAKE")
            fake_frame_ratio = fake_frame_count / len(frames_with_faces) if frames_with_faces else 0.0
            
            # Video verdict: Flag as DEEPFAKE if mean probability exceeds threshold OR >=35% frames show manipulation
            is_deepfake = (mean_fake_prob >= self.inference_service.threshold) or (fake_frame_ratio >= 0.35)
            overall_prediction = "DEEPFAKE" if is_deepfake else "REAL"
            
            if is_deepfake:
                display_deepfake_prob = max(mean_fake_prob, fake_frame_ratio)
            else:
                display_deepfake_prob = mean_fake_prob
            
            if len(frames_with_faces) < extracted_count:
                warnings.append(f"{extracted_count - len(frames_with_faces)} frame(s) had no detectable face and were excluded from scoring.")
            
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
                "video_duration_seconds": duration,
                "video_fps": round(video_fps, 2),
                "video_resolution": [width, height],
                "warnings": warnings,
            }
        
        finally:
            if cap is not None:
                cap.release()
            if created_temp_file and tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _analyze_single_frame(self, frame: np.ndarray, count: int, frame_idx: int, video_fps: float) -> dict:
        """Analyze a single video frame with resolution scaling for lightweight, memory-safe execution."""
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        
        # Scale down to max 480px: MTCNN detects faces accurately at this resolution while saving 80% RAM & CPU
        if max_dim > 480:
            scale = 480.0 / max_dim
            sw, sh = int(w * scale), int(h * scale)
            small_frame = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
            det_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        else:
            det_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        pil_for_det = Image.fromarray(det_rgb)
        detections = detect_faces(pil_for_det, self.inference_service.face_detector)
        
        timestamp = round(frame_idx / video_fps, 2) if video_fps > 0 else 0.0
        frame_result = {
            "frame_index": count + 1,
            "timestamp_seconds": timestamp,
            "face_detected": len(detections) > 0,
            "face_count": len(detections),
        }
        
        if len(detections) > 0:
            selected = select_face(detections, strategy="largest")
            face_np = crop_face(det_rgb, selected.box, margin=0.2)
            face_pil = Image.fromarray(face_np)
            
            input_tensor = self.inference_service.transform(face_pil).unsqueeze(0).to(self.inference_service.device)
            with torch.no_grad():
                outputs = self.inference_service.model(input_tensor)
                prob_deepfake = torch.sigmoid(outputs).item()
            
            frame_result["deepfake_probability"] = round(prob_deepfake, 4)
            frame_result["real_probability"] = round(1.0 - prob_deepfake, 4)
            frame_result["prediction"] = "DEEPFAKE" if prob_deepfake >= self.inference_service.threshold else "REAL"
            del input_tensor, outputs, face_pil, face_np
        else:
            input_tensor = self.inference_service.transform(pil_for_det).unsqueeze(0).to(self.inference_service.device)
            with torch.no_grad():
                outputs = self.inference_service.model(input_tensor)
                prob_fallback = torch.sigmoid(outputs).item()
            
            frame_result["fallback_probability"] = round(prob_fallback, 4)
            frame_result["deepfake_probability"] = None
            frame_result["real_probability"] = None
            frame_result["prediction"] = None
            del input_tensor, outputs
        
        del pil_for_det, det_rgb
        return frame_result
