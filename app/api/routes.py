import logging
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, status
from typing import Dict, Any

from app.schemas.prediction import PredictionResponse, HealthResponse, ErrorResponse, VideoPredictionResponse
from app.services.inference_service import InferenceService
from app.services.video_inference_service import VideoInferenceService
from app.config import get_app_config, AppConfig

logger = logging.getLogger(__name__)
router = APIRouter()

def get_inference_service() -> InferenceService:
    """Dependency injection for InferenceService."""
    return InferenceService()


async def validate_uploaded_file(file: UploadFile, config: AppConfig) -> bytes:
    """Validate uploaded file type and size."""
    if file.content_type not in config.allowed_mime_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type. Allowed: {', '.join(config.allowed_mime_types)}"
        )
        
    contents = await file.read()
    if len(contents) > config.max_upload_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size allowed is {config.max_upload_size // (1024*1024)}MB."
        )
        
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )
        
    return contents


@router.get("/health", response_model=HealthResponse)
async def health_check(
    service: InferenceService = Depends(get_inference_service)
):
    """Check API health and model status."""
    is_loaded = service.is_model_loaded
    return HealthResponse(
        status="ok",
        model_loaded=is_loaded,
        model_name="EfficientNet-B0" if is_loaded else None,
        device=str(service.device),
        message="Service is fully operational." if is_loaded else "Model failed to load."
    )


@router.post(
    "/predict", 
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def predict(
    file: UploadFile = File(...),
    service: InferenceService = Depends(get_inference_service),
    config: AppConfig = Depends(get_app_config)
):
    """Run model inference on uploaded image."""
    try:
        if not service.is_model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded."
            )
            
        file_bytes = await validate_uploaded_file(file, config)
        result = service.predict(file_bytes)
        
        # Remove internal fields not meant for response
        if "input_tensor" in result:
            del result["input_tensor"]
        if "face_np" in result:
            del result["face_np"]
            
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        logger.error(f"Predict endpoint error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post("/explain")
async def explain(
    file: UploadFile = File(...),
    service: InferenceService = Depends(get_inference_service),
    config: AppConfig = Depends(get_app_config)
):
    """Generate explainability visualizations for uploaded image."""
    try:
        if not service.is_model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded."
            )
            
        file_bytes = await validate_uploaded_file(file, config)
        result = service.explain(file_bytes)
        
        if result.get("error"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
            
        return {
            "heatmap": result["heatmap"],
            "overlay": result["overlay"]
        }
        
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        logger.error(f"Explain endpoint error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post(
    "/predict-and-explain", 
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def predict_and_explain(
    file: UploadFile = File(...),
    service: InferenceService = Depends(get_inference_service),
    config: AppConfig = Depends(get_app_config)
):
    """Run model inference and generate explainability visualizations in one call."""
    try:
        if not service.is_model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded."
            )
            
        file_bytes = await validate_uploaded_file(file, config)
        
        # Save for debugging
        debug_dir = Path("data/debug_uploads")
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_file = debug_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(debug_file, "wb") as df:
            df.write(file_bytes)
            
        result = service.predict_and_explain(file_bytes)
        logger.info(f"[IMAGE PREDICTION] File: {file.filename} -> Verdict: {result.get('prediction')} | Real: {result.get('real_probability', 0)*100:.1f}% | Fake: {result.get('deepfake_probability', 0)*100:.1f}% | Face Detected: {result.get('face_detected')}")
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        logger.error(f"Predict and explain endpoint error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}")


@router.post(
    "/predict-video",
    response_model=VideoPredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def predict_video(
    file: UploadFile = File(...),
    config: AppConfig = Depends(get_app_config)
):
    """Run deepfake detection on an uploaded video file."""
    try:
        # Validate video file type
        video_mime_types = ["video/mp4", "video/avi", "video/x-msvideo", "video/quicktime", 
                           "video/x-matroska", "video/webm", "application/octet-stream"]
        if file.content_type not in video_mime_types:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type '{file.content_type}'. Please upload a video file (MP4, AVI, MOV, MKV, WEBM)."
            )
        
        # Read video bytes (limit to 100MB)
        max_video_size = 100 * 1024 * 1024
        contents = await file.read()
        if len(contents) > max_video_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Video file too large. Maximum size is 100MB."
            )
        
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )
        
        video_service = VideoInferenceService()
        
        if not video_service.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded."
            )
        
        result = video_service.process_video(contents, max_frames=30, extraction_fps=1.0)
        logger.info(f"[VIDEO PREDICTION] File: {file.filename} -> Verdict: {result.get('prediction')} | Real: {result.get('real_probability', 0)*100:.1f}% | Fake: {result.get('deepfake_probability', 0)*100:.1f}% | Frames Analyzed: {result.get('frames_analyzed')}")
        return VideoPredictionResponse(**result)
        
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        logger.error(f"Video predict endpoint error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Video analysis failed: {str(e)}")
