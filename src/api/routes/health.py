"""
Health Check Routes
"""

from fastapi import APIRouter
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
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        model_loaded=True
    )


@router.get("/readiness")
async def readiness_check():
    """Readiness check for Kubernetes."""
    # Check if model is loaded, database is accessible, etc.
    return {"status": "ready"}


@router.get("/liveness")
async def liveness_check():
    """Liveness check for Kubernetes."""
    return {"status": "alive"}
