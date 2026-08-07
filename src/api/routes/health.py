"""
Health Check Routes
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import logging

from ...contracts import PUBLIC_API_VERSION

logger = logging.getLogger(__name__)

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    model_loaded: bool
    model_version: str | None = None
    model_sha256: str | None = None


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check endpoint."""
    model_loaded = bool(getattr(request.app.state, "model_loaded", False))
    model = getattr(request.app.state, "model", None)
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        timestamp=datetime.now().isoformat(),
        version=PUBLIC_API_VERSION,
        model_loaded=model_loaded,
        model_version=model.metadata["model_version"] if model else None,
        model_sha256=model.model_sha256 if model else None,
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
    return {
        "status": "ready",
        "model_version": request.app.state.model.metadata["model_version"],
        "model_sha256": request.app.state.model.model_sha256,
    }


@router.get("/liveness")
async def liveness_check():
    """Liveness check for Kubernetes."""
    return {"status": "alive"}
