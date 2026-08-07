"""
FastAPI Main Application
"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
import uuid
from pathlib import Path

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from .routes import predict, models, health
from .middleware import RateLimitMiddleware
from ..contracts import PUBLIC_API_NAME, PUBLIC_API_VERSION
from ..utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
METRICS_REGISTRY = CollectorRegistry()
REQUESTS = Counter(
    "wafer_classifier_requests_total",
    "Wafer classifier requests",
    ("endpoint", "status"),
    registry=METRICS_REGISTRY,
)
LATENCY = Histogram(
    "wafer_classifier_request_duration_seconds",
    "Wafer classifier request latency",
    ("endpoint",),
    registry=METRICS_REGISTRY,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    logger.info("Starting %s...", PUBLIC_API_NAME)
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
    except (FileNotFoundError, ValueError) as error:
        app.state.model = None
        app.state.startup_warning = (
            f"Confirmed model is unavailable: {error}. API started in degraded mode."
        )
        logger.warning(app.state.startup_warning)

    yield

    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title=PUBLIC_API_NAME,
    version=PUBLIC_API_VERSION,
    description="Hash-verified synthetic wafer-pattern classification reference",
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
app.add_middleware(RateLimitMiddleware)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.time()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(time.time() - start_time)
        return response
    finally:
        elapsed = time.time() - start_time
        REQUESTS.labels(endpoint=request.url.path, status=str(status_code)).inc()
        LATENCY.labels(endpoint=request.url.path).observe(elapsed)


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
        "name": PUBLIC_API_NAME,
        "version": PUBLIC_API_VERSION,
        "status": "running" if app.state.model_loaded else "degraded",
        "model_loaded": app.state.model_loaded,
        "startup_warning": app.state.startup_warning,
        "docs": "/docs",
        "health": "/api/v1/health",
        "confirmed_endpoint": "/api/v1/classify-image",
        "boundary": "Synthetic image classification only; STDF parsing and yield prediction are not confirmed",
    }


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(METRICS_REGISTRY), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
