"""
Health Check Routes
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    model_loaded: bool


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check endpoint."""
    model_loaded = bool(getattr(request.app.state, "model_loaded", False))
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        model_loaded=model_loaded
    )


@router.get("/readiness")
async def readiness_check(request: Request):
    """Readiness check for local deployments."""
    model_loaded = bool(getattr(request.app.state, "model_loaded", False))
    if not model_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": getattr(request.app.state, "startup_warning", "Model unavailable")
            }
        )
    return {"status": "ready"}


@router.get("/liveness")
async def liveness_check():
    """Liveness check for Kubernetes."""
    return {"status": "alive"}
