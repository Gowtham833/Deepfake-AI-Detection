import os
import io
import cv2
import torch
import numpy as np
from PIL import Image
import logging
from torchvision import transforms

from app.config import get_app_config
from app.schemas.prediction import PredictionResponse
from src.utils.device import get_device
from app.services.gradcam_service import GradCAMService
from src.models.efficientnet import create_model
from src.data.face_detection import FaceDetector, detect_faces, select_face, crop_face

logger = logging.getLogger(__name__)

REAL = 0
DEEPFAKE = 1
CLASS_NAMES = ["REAL", "DEEPFAKE"]

class InferenceService:
    """Singleton service for model inference."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InferenceService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
            
        self.config = get_app_config()
        self.device = get_device(self.config.device)
        self.model = None
        self.face_detector = None
        self.transform = None
        self.gradcam_service = GradCAMService()
        self.threshold = 0.4  # Lowered for better fake detection sensitivity
        
        self._load_dependencies()
        self._initialized = True
        
    def _load_dependencies(self):
        """Load FaceDetector and the classification model."""
        try:
            # Initialize MTCNN for face detection
            self.face_detector = FaceDetector(device=self.device)
            
            # Setup image transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            # Initialize model using the correct project module
            self.model = create_model({"model": {"pretrained": False}})
            
            # Load checkpoint if it exists
            checkpoint_path = self.config.model_checkpoint_path
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                # Set threshold: prefer env var, then checkpoint metadata, then config, default to 0.5
                thresh_env = os.environ.get("DETECTION_THRESHOLD")
                if thresh_env:
                    self.threshold = float(thresh_env)
                elif isinstance(checkpoint, dict) and "threshold" in checkpoint:
                    self.threshold = float(checkpoint["threshold"])
                elif hasattr(self.config, "threshold"):
                    self.threshold = float(self.config.threshold)
                else:
                    self.threshold = 0.5
                logger.info(f"Using detection threshold: {self.threshold}")
                
                logger.info(f"Model loaded successfully from {checkpoint_path}")
            else:
                logger.warning(f"No checkpoint found at {checkpoint_path}. Application will start but predictions will be invalid without a checkpoint.")
                self.model = None # Set to None so predict() correctly identifies it's not loaded
                return
                
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model dependencies: {e}")
            self.model = None
            
    @property
    def is_model_loaded(self) -> bool:
        return self.model is not None

    def _process_image_bytes(self, image_bytes: bytes):
        """Decode image bytes and detect faces."""
        try:
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image")
                
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Detect faces
            detections = detect_faces(pil_img, self.face_detector)
            face_count = len(detections)
            
            if face_count == 0:
                return None, None, img_rgb, []
                
            # Select best face using strategy
            selected_face = select_face(detections, strategy="largest")
            best_box = selected_face.box
            
            # Crop face using consistent preprocessing
            face_np = crop_face(img_rgb, best_box, margin=0.2)
            face_pil = Image.fromarray(face_np)
            
            all_boxes = [d.box for d in detections]
            
            return face_pil, best_box, face_np, all_boxes
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise ValueError(f"Image processing failed: {str(e)}")

    def predict(self, image_bytes: bytes) -> dict:
        """Run inference on image bytes, return prediction without explanations."""
        if not self.is_model_loaded:
            raise RuntimeError("Model is not loaded")
            
        warnings = []
        try:
            face_pil, best_box, face_np, all_boxes = self._process_image_bytes(image_bytes)
            
            face_detected = best_box is not None
            face_count = len(all_boxes)
            
            if not face_detected:
                # Fallback: use the whole image resized to 224x224
                warnings.append("No face detected — analyzing the whole image as fallback.")
                
                nparr = np.frombuffer(image_bytes, np.uint8)
                import cv2 as _cv2
                img = _cv2.imdecode(nparr, _cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Could not decode image")
                img_rgb = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
                face_np = img_rgb
                face_pil = Image.fromarray(img_rgb)

            # Preprocess for model
            input_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                # Single logit output for binary classification, use sigmoid
                prob_deepfake = torch.sigmoid(outputs).item()
                
            real_prob = 1.0 - prob_deepfake
            deepfake_prob = prob_deepfake
            
            prediction = "DEEPFAKE" if deepfake_prob >= self.threshold else "REAL"
            
            return {
                "prediction": prediction,
                "real_probability": real_prob,
                "deepfake_probability": deepfake_prob,
                "threshold": self.threshold,
                "model_name": "EfficientNet-B0",
                "face_detected": face_detected,
                "face_count": face_count,
                "selected_face_box": [int(v) for v in best_box] if best_box is not None else None,
                "explanation_available": False,
                "warnings": warnings,
                "input_tensor": input_tensor,
                "face_np": face_np
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def explain(self, image_bytes: bytes) -> dict:
        """Generate only explanations for the image."""
        try:
            pred_data = self.predict(image_bytes)
            
            if not pred_data.get("face_detected"):
                return {"heatmap": None, "overlay": None, "error": "No face detected"}
                
            input_tensor = pred_data.pop("input_tensor")
            face_np = pred_data.pop("face_np")
            
            # Requires gradients for Grad-CAM
            input_tensor.requires_grad = True
            self.model.eval() # Keep in eval mode
            
            explanation = self.gradcam_service.generate_explanation(
                self.model, input_tensor, face_np, {}, self.device
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Explanation error: {e}")
            return {"heatmap": None, "overlay": None, "error": str(e)}

    def predict_and_explain(self, image_bytes: bytes) -> dict:
        """Run inference and generate explanations in one pass."""
        try:
            pred_data = self.predict(image_bytes)
            
            input_tensor = pred_data.pop("input_tensor", None)
            face_np = pred_data.pop("face_np", None)
            
            if not pred_data.get("face_detected"):
                pred_data["processed_face_image"] = None
                pred_data["gradcam_heatmap"] = None
                pred_data["gradcam_overlay"] = None
                pred_data["explanation_available"] = False
                return pred_data
            
            # Base64 encode the processed face
            processed_face_b64 = self.gradcam_service.image_to_base64(face_np)
            pred_data["processed_face_image"] = processed_face_b64
            
            # Generate explanation
            if self.gradcam_service.available:
                input_tensor.requires_grad = True
                
                # Make sure parameters require gradients for Grad-CAM
                for param in self.model.parameters():
                    param.requires_grad = True
                    
                explanation = self.gradcam_service.generate_explanation(
                    self.model, input_tensor, face_np, {}, self.device
                )
                
                if not explanation.get("error"):
                    pred_data["gradcam_heatmap"] = explanation["heatmap"]
                    pred_data["gradcam_overlay"] = explanation["overlay"]
                    pred_data["explanation_available"] = True
                else:
                    pred_data["warnings"].append(f"Grad-CAM failed: {explanation.get('error')}")
            else:
                pred_data["warnings"].append("Grad-CAM service is not available.")
                
            return pred_data
            
        except Exception as e:
            logger.error(f"Predict and explain error: {e}")
            raise RuntimeError(f"Failed to generate prediction and explanation: {str(e)}")
