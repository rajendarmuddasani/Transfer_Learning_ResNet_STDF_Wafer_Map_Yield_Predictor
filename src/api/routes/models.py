"""Read-only confirmed model registry routes."""

from fastapi import APIRouter, Depends, HTTPException

from ..inference import get_model
from ..security import require_api_key


router = APIRouter()


async def confirmed_model_record() -> dict:
    engine = get_model()
    metrics = engine.evaluation["confirmation_metrics"]
    return {
        "model_id": engine.metadata["model_version"],
        "model_name": "Public Synthetic ResNet-18 Classifier",
        "architecture": engine.metadata["architecture"],
        "version": engine.metadata["model_version"],
        "stage": "CONFIRMED_SYNTHETIC",
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "minimum_class_recall": metrics["minimum_class_recall"],
        "model_sha256": engine.model_sha256,
        "data_scope": engine.evaluation["data_scope"],
    }


@router.get("/models", dependencies=[Depends(require_api_key)])
async def list_models():
    return [await confirmed_model_record()]


@router.get("/models/{model_id}", dependencies=[Depends(require_api_key)])
async def get_model_info(model_id: str):
    record = await confirmed_model_record()
    if model_id != record["model_id"]:
        raise HTTPException(status_code=404, detail="Model not found")
    return record


@router.post("/models/{model_id}/promote")
async def unsupported_promotion(model_id: str):
    raise HTTPException(
        status_code=501,
        detail=f"Automated model promotion is not implemented: {model_id}",
    )
