import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router as api_router
from app.services.inference_service import InferenceService
from src.utils.io import get_project_root

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Deepfake Detection API",
        description="API for detecting deepfakes using EfficientNet and Grad-CAM",
        version="1.0.0",
    )
    
    # Configure CORS (allow all for development)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API router
    app.include_router(api_router, prefix="/api")
    
    # Mount static files and templates
    root_dir = get_project_root()
    app_dir = os.path.join(root_dir, "app")
    static_dir = os.path.join(app_dir, "static")
    templates_dir = os.path.join(app_dir, "templates")
    
    # Ensure directories exist
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(os.path.join(static_dir, "css"), exist_ok=True)
    os.makedirs(os.path.join(static_dir, "js"), exist_ok=True)
    os.makedirs(templates_dir, exist_ok=True)
    
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    @app.get("/", include_in_schema=False)
    async def index():
        """Serve the frontend index.html page."""
        index_path = os.path.join(templates_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Frontend index.html not found"}

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        """Serve app favicon."""
        fav_path = os.path.join(static_dir, "icons", "favicon.png")
        if os.path.exists(fav_path):
            return FileResponse(fav_path, media_type="image/png")
        from fastapi.responses import Response
        return Response(status_code=204)

    @app.get("/manifest.json", include_in_schema=False)
    async def pwa_manifest():
        """Serve PWA Web App Manifest for Android and browser installability."""
        manifest_path = os.path.join(static_dir, "manifest.json")
        if os.path.exists(manifest_path):
            return FileResponse(manifest_path, media_type="application/manifest+json")
        from fastapi.responses import Response
        return Response(status_code=404)

    @app.get("/sw.js", include_in_schema=False)
    async def service_worker():
        """Serve PWA Service Worker with root scope permissions."""
        sw_path = os.path.join(static_dir, "sw.js")
        if os.path.exists(sw_path):
            return FileResponse(
                sw_path,
                media_type="application/javascript",
                headers={"Service-Worker-Allowed": "/"}
            )
        from fastapi.responses import Response
        return Response(status_code=404)
        
    @app.on_event("startup")
    async def startup_event():
        """Execute logic on application startup."""
        logger.info("Starting up Deepfake Detection Web Application...")
        
        # Initialize InferenceService singleton (loads model)
        service = InferenceService()
        if service.is_model_loaded:
            logger.info("Model loaded successfully during startup.")
        else:
            logger.warning("Failed to load model during startup.")
            
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
