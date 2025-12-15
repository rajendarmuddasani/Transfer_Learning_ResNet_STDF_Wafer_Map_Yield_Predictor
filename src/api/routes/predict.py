"""
Prediction Routes
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import tempfile
import logging
from pathlib import Path
import numpy as np

from .inference import predict_wafer, batch_predict
from ..data import parse_stdf_file, generate_wafer_map

logger = logging.getLogger(__name__)

router = APIRouter()


class PredictionRequest(BaseModel):
    """Request model for prediction."""
    wafer_id: Optional[str] = None
    product_id: Optional[str] = Field(None, description="Product ID for model selection")
    test_completion_pct: Optional[float] = Field(10.0, ge=0, le=100)
    include_gradcam: bool = Field(False, description="Generate Grad-CAM heatmap")
    gradcam_layer: str = Field("layer4", description="Layer for Grad-CAM")


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    wafer_id: str
    prediction: dict
    model_version: str
    inference_time_ms: float
    grad_cam_url: Optional[str] = None
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    wafer_ids: Optional[List[str]] = None
    lot_id: Optional[str] = None
    include_gradcam: bool = False
    model_version: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    job_id: str
    status: str
    total_wafers: int
    estimated_time_seconds: int
    status_url: str


@router.post("/predict", response_model=PredictionResponse)
async def predict_single(
    stdf_file: Optional[UploadFile] = File(None),
    wafer_map_image: Optional[UploadFile] = File(None),
    product_id: Optional[str] = None,
    test_completion_pct: float = 10.0,
    include_gradcam: bool = False,
    gradcam_layer: str = "layer4"
):
    """
    Predict yield for a single wafer.
    
    Accepts either STDF file or wafer map image.
    """
    if not stdf_file and not wafer_map_image:
        raise HTTPException(
            status_code=400,
            detail="Either stdf_file or wafer_map_image must be provided"
        )
    
    try:
        # Process input
        if stdf_file:
            # Save STDF file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".stdf") as tmp:
                content = await stdf_file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            # Parse STDF and generate wafer map
            logger.info(f"Parsing STDF file: {stdf_file.filename}")
            wafer_data = parse_stdf_file(tmp_path)
            
            # Generate wafer map image
            wafer_map_array = generate_wafer_map(wafer_data)
            wafer_id = wafer_data.wafer_id
            
            # Clean up temp file
            Path(tmp_path).unlink()
            
        else:
            # Process wafer map image directly
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                content = await wafer_map_image.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            from PIL import Image
            wafer_map_array = np.array(Image.open(tmp_path))
            wafer_id = wafer_map_image.filename.split('.')[0]
            
            # Clean up temp file
            Path(tmp_path).unlink()
        
        # Run inference
        logger.info(f"Running inference for wafer: {wafer_id}")
        result = await predict_wafer(
            wafer_map_array,
            wafer_id=wafer_id,
            include_gradcam=include_gradcam,
            gradcam_layer=gradcam_layer
        )
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit batch prediction job.
    
    Process multiple wafers asynchronously.
    """
    try:
        # Validate request
        if not request.wafer_ids and not request.lot_id:
            raise HTTPException(
                status_code=400,
                detail="Either wafer_ids or lot_id must be provided"
            )
        
        # Generate job ID
        import uuid
        from datetime import datetime
        job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Get wafer list
        if request.lot_id:
            # Query database for wafers in lot
            wafer_ids = await get_wafers_in_lot(request.lot_id)
        else:
            wafer_ids = request.wafer_ids
        
        total_wafers = len(wafer_ids)
        estimated_time = total_wafers * 2  # 2 seconds per wafer estimate
        
        # Add batch job to background tasks
        background_tasks.add_task(
            batch_predict,
            job_id=job_id,
            wafer_ids=wafer_ids,
            include_gradcam=request.include_gradcam,
            model_version=request.model_version
        )
        
        logger.info(f"Batch job submitted: {job_id}, {total_wafers} wafers")
        
        return BatchPredictionResponse(
            job_id=job_id,
            status="QUEUED",
            total_wafers=total_wafers,
            estimated_time_seconds=estimated_time,
            status_url=f"/api/v1/jobs/{job_id}"
        )
        
    except Exception as e:
        logger.error(f"Batch prediction submission failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Get status of batch prediction job.
    """
    try:
        # Query job status from database/cache
        # Placeholder implementation
        return {
            "job_id": job_id,
            "status": "COMPLETED",
            "progress": {
                "completed": 10,
                "total": 10,
                "percentage": 100.0
            },
            "results_url": f"/api/v1/results/{job_id}",
            "created_at": "2025-12-06T10:30:00Z",
            "completed_at": "2025-12-06T10:35:00Z"
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@router.get("/results/{job_id}")
async def get_batch_results(job_id: str):
    """
    Get results of completed batch prediction job.
    """
    try:
        # Retrieve results from database
        # Placeholder implementation
        return {
            "job_id": job_id,
            "status": "COMPLETED",
            "results": [
                {
                    "wafer_id": "W001",
                    "yield_pred": 87.3,
                    "defect_class": "EdgeEffect",
                    "confidence": 0.92
                }
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Results not found: {job_id}")


async def get_wafers_in_lot(lot_id: str) -> List[str]:
    """Helper function to get wafer IDs in a lot."""
    # Placeholder - query database
    return [f"{lot_id}-W{i:03d}" for i in range(1, 11)]
