"""Bounded image-classification routes for the confirmed public model."""

from __future__ import annotations

import io
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from ..inference import get_model
from ..security import require_api_key


router = APIRouter()
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg"}


class ClassificationResponse(BaseModel):
    wafer_reference: str
    task: str
    defect_class: str
    defect_probabilities: dict[str, float]
    confidence: float
    model_version: str
    model_sha256: str
    inference_time_ms: float
    request_id: str
    timestamp: str


@router.post(
    "/classify-image",
    response_model=ClassificationResponse,
    dependencies=[Depends(require_api_key)],
)
async def classify_image(request: Request, wafer_map_image: UploadFile = File(...)):
    if wafer_map_image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PNG and JPEG wafer-map images are accepted",
        )
    content = await wafer_map_image.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MiB limit")
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=422, detail="Invalid image payload") from error

    result = get_model().predict(image)
    filename = wafer_map_image.filename or "wafer-map"
    wafer_reference = filename.rsplit(".", 1)[0][:100]
    return ClassificationResponse(
        wafer_reference=wafer_reference,
        task="synthetic wafer-pattern classification",
        defect_class=result["defect_class"],
        defect_probabilities=result["probabilities"],
        confidence=result["confidence"],
        model_version=result["model_version"],
        model_sha256=result["model_sha256"],
        inference_time_ms=result["inference_time_ms"],
        request_id=request.state.request_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.post("/predict")
async def unsupported_legacy_predict(stdf_file: UploadFile | None = File(None)):
    detail = (
        "STDF parsing is outside the confirmed public model boundary"
        if stdf_file is not None
        else "Use /api/v1/classify-image with a PNG or JPEG wafer map"
    )
    raise HTTPException(status_code=501, detail=detail)


@router.post("/predict/batch")
async def unsupported_batch():
    raise HTTPException(status_code=501, detail="Batch classification is not implemented")


@router.get("/jobs/{job_id}")
async def unsupported_job_status(job_id: str):
    raise HTTPException(status_code=501, detail=f"Job tracking is not implemented: {job_id}")


@router.get("/results/{job_id}")
async def unsupported_results(job_id: str):
    raise HTTPException(status_code=501, detail=f"Result storage is not implemented: {job_id}")
