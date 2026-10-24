import io
import base64
import logging
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class GradCAMService:
    """Service wrapper for Grad-CAM explainability."""
    
    def __init__(self):
        try:
            from src.explainability.gradcam import GradCAM
            self.gradcam_class = GradCAM
            self.available = True
        except ImportError as e:
            logger.warning(f"Could not import GradCAM: {e}")
            self.gradcam_class = None
            self.available = False
            
    def image_to_base64(self, image_array: np.ndarray) -> str:
        """Convert a numpy image array (RGB) to base64 PNG string."""
        if image_array is None:
            return None
        
        # Ensure image is in 0-255 range and uint8
        if image_array.dtype != np.uint8:
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)
                
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Encode to PNG
        success, buffer = cv2.imencode(".png", bgr_image)
        if not success:
            logger.error("Failed to encode image to PNG format")
            return None
            
        # Convert to base64
        return base64.b64encode(buffer).decode("utf-8")
        
    def generate_explanation(self, model, input_tensor, face_image: np.ndarray, config, device) -> dict:
        """Generate Grad-CAM heatmap and overlay."""
        if not self.available or input_tensor is None:
            return {"heatmap": None, "overlay": None, "error": "Grad-CAM not available or invalid input"}
            
        try:
            # Directly use the last EfficientNet feature block — correct Grad-CAM target layer
            if hasattr(model, 'features'):
                target_layer = model.features[-1]
            else:
                raise RuntimeError("Cannot determine Grad-CAM target layer: model has no 'features' attribute.")
                
            gradcam = self.gradcam_class(model, target_layer)
            
            # Generate heatmap
            heatmap = gradcam.generate(input_tensor)
            
            # If the output is a batch, take the first one
            if len(heatmap.shape) == 3:
                heatmap = heatmap[0]
                
            # Resize heatmap to match face image size
            heatmap_resized = cv2.resize(heatmap, (face_image.shape[1], face_image.shape[0]))
            
            # Create color map
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Create overlay
            alpha = 0.5
            overlay = cv2.addWeighted(face_image, alpha, heatmap_colored, 1 - alpha, 0)
            
            return {
                "heatmap": self.image_to_base64(heatmap_colored),
                "overlay": self.image_to_base64(overlay),
                "error": None
            }
        except Exception as e:
            logger.error(f"Error generating Grad-CAM explanation: {e}")
            import traceback
            traceback.print_exc()
            return {"heatmap": None, "overlay": None, "error": str(e)}
