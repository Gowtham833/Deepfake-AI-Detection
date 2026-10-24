import os
from typing import List
from pydantic import BaseModel, Field

from src.utils.io import load_config, get_project_root


class AppConfig(BaseModel):
    """Application configuration schema."""
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_size: int = 10 * 1024 * 1024  # 10 MB
    allowed_mime_types: List[str] = ["image/jpeg", "image/png", "image/webp"]
    allowed_extensions: List[str] = [".jpg", ".jpeg", ".png", ".webp"]
    max_image_dimension: int = 4096
    model_checkpoint_path: str = ""
    device: str = "cuda"
    max_video_upload_size: int = 100 * 1024 * 1024  # 100 MB
    allowed_video_mime_types: List[str] = ["video/mp4", "video/avi", "video/x-msvideo", "video/quicktime", "video/x-matroska", "video/webm"]
    allowed_video_extensions: List[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

def get_app_config() -> AppConfig:
    """Load configuration from central config and environment."""
    root = get_project_root()
    config_path = os.path.join(root, "configs", "config.yaml")
    
    # Try to load yaml config, fallback to defaults if not found
    try:
        yaml_config = load_config(config_path)
    except FileNotFoundError:
        yaml_config = {}
        
    api_config = yaml_config.get("api", {})
    model_config = yaml_config.get("model", {})
    
    # Checkpoint path: check env var, user's root folder, or default models dir
    candidates = [
        os.environ.get("MODEL_CHECKPOINT_PATH"),
        str(root.parent / "deepfake_best_model.pth"),
        str(root / "deepfake_best_model.pth"),
        str(root / "models" / "best_model.pth"),
        str(root / "models" / "deepfake_best_model.pth"),
        model_config.get("best_model_path"),
    ]
    checkpoint_path = ""
    for cand in candidates:
        if cand and os.path.exists(cand):
            checkpoint_path = cand
            break
    if not checkpoint_path:
        checkpoint_path = os.environ.get(
            "MODEL_CHECKPOINT_PATH", 
            model_config.get("best_model_path", os.path.join(root, "models", "best_model.pth"))
        )
    
    return AppConfig(
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        max_upload_size=api_config.get("max_upload_size", 10 * 1024 * 1024),
        allowed_mime_types=api_config.get("allowed_mime_types", ["image/jpeg", "image/png", "image/webp"]),
        allowed_extensions=api_config.get("allowed_extensions", [".jpg", ".jpeg", ".png", ".webp"]),
        max_image_dimension=api_config.get("max_image_dimension", 4096),
        model_checkpoint_path=checkpoint_path,
        device=yaml_config.get("device", "auto")
    )

app_config = get_app_config()
