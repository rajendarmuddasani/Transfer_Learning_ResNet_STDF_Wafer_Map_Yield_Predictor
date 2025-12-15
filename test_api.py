"""
Simple API test without full dependencies
"""

import sys
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Create minimal app
app = FastAPI(
    title="P02 Yield Predictor API",
    description="Wafer defect classification using transfer learning",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "P02 Yield Predictor API", "status": "healthy"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "p02-yield-predictor",
        "version": "1.0.0"
    }

@app.get("/api/v1/health")
async def api_health():
    """API health check."""
    return {
        "status": "ok",
        "database": "not_connected",
        "redis": "not_connected",
        "models": []
    }

@app.post("/api/v1/predict")
async def predict(
    file: UploadFile = File(...),
    wafer_id: Optional[str] = Form(None)
):
    """Mock prediction endpoint."""
    # Mock prediction response
    defect_classes = ['Normal', 'EdgeEffect', 'CenterCluster', 'RingPattern', 
                      'QuadrantFailure', 'Scratch', 'RandomFailure', 'MixedMode']
    
    predicted_class = random.choice(defect_classes)
    probabilities = {cls: random.random() for cls in defect_classes}
    # Normalize probabilities
    total = sum(probabilities.values())
    probabilities = {k: v/total for k, v in probabilities.items()}
    
    return {
        "wafer_id": wafer_id or f"W{random.randint(1000, 9999)}",
        "prediction": {
            "yield": round(random.uniform(0.75, 0.98), 4),
            "defect_class": predicted_class,
            "defect_probabilities": probabilities,
            "confidence": round(random.uniform(0.85, 0.99), 4),
            "uncertainty": round(random.uniform(0.01, 0.15), 4)
        },
        "model_version": "resnet18-v1.0.0",
        "timestamp": "2025-12-06T15:30:00Z",
        "processing_time_ms": random.randint(50, 200)
    }

@app.get("/api/v1/models")
async def list_models():
    """Mock models endpoint."""
    return {
        "models": [
            {
                "model_id": "resnet18-v1.0.0",
                "architecture": "resnet18",
                "status": "production",
                "accuracy": 0.94,
                "created_at": "2025-12-01T10:00:00Z"
            },
            {
                "model_id": "resnet50-v1.1.0",
                "architecture": "resnet50",
                "status": "staging",
                "accuracy": 0.96,
                "created_at": "2025-12-05T14:30:00Z"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting P02 Yield Predictor API on http://localhost:8001")
    print("Press CTRL+C to quit")
    uvicorn.run(app, host="0.0.0.0", port=8001)
