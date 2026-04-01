"""
FastAPI Main Application
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
from pathlib import Path

from .routes import predict, models, health
from .middleware import rate_limiter
from ..utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    logger.info("Starting Transfer_Learning_ResNet_STDF_Wafer_Map_Yield_Predictor API...")
    config = load_config("config/api_config.yaml")
    app.state.config = config
    app.state.model_loaded = False
    app.state.startup_warning = None
    
    # Load model on startup
    from .inference import load_model
    model_path = Path(config["model"]["production_model_path"])

    try:
        app.state.model = load_model(str(model_path))
        app.state.model_loaded = True
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        app.state.model = None
        app.state.startup_warning = (
            f"Model artifact not found at {model_path}. API started in degraded mode."
        )
        logger.warning(app.state.startup_warning)
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="Transfer_Learning_ResNet_STDF_Wafer_Map_Yield_Predictor API",
    version="1.0.0",
    description="REST API for semiconductor wafer yield prediction using ResNet transfer learning",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "path": request.url.path
        }
    )


# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(models.router, prefix="/api/v1", tags=["Models"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Transfer_Learning_ResNet_STDF_Wafer_Map_Yield_Predictor API",
        "version": "1.0.0",
        "status": "running" if app.state.model_loaded else "degraded",
        "model_loaded": app.state.model_loaded,
        "startup_warning": app.state.startup_warning,
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
