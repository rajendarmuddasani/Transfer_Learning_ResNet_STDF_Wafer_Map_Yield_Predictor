"""
Model Inference Module
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Optional, Dict, List
import time
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelInference:
    """Handle model inference with caching and optimization."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model checkpoint or ONNX file
            device: Device to run inference on
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.transform = self._get_transform()
        
        logger.info(f"Initializing inference on device: {device}")
    
    def load_model(self):
        """Load model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load model based on file extension
        if self.model_path.suffix == ".onnx":
            self._load_onnx_model()
        else:
            self._load_pytorch_model()
        
        logger.info(f"Model loaded from {self.model_path}")
    
    def _load_pytorch_model(self):
        """Load PyTorch model."""
        from ..models import create_model
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Create model
        self.model = create_model(
            architecture=checkpoint.get('architecture', 'resnet18'),
            num_classes=checkpoint.get('num_classes', 8),
            task='classification',
            pretrained=False
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _load_onnx_model(self):
        """Load ONNX model."""
        import onnxruntime as ort
        
        # Create ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.model = ort.InferenceSession(str(self.model_path), providers=providers)
        logger.info(f"ONNX Runtime providers: {self.model.get_providers()}")
    
    def _get_transform(self):
        """Get image preprocessing transform."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Preprocessed tensor (1, 3, 224, 224)
        """
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        # Apply transform
        tensor = self.transform(image)
        
        # Add batch dimension
        return tensor.unsqueeze(0)
    
    def predict(
        self,
        image: np.ndarray,
        return_probabilities: bool = True
    ) -> Dict:
        """
        Run inference on image.
        
        Args:
            image: RGB image array
            return_probabilities: Return class probabilities
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            self.load_model()
        
        start_time = time.time()
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Run inference
        with torch.no_grad():
            if isinstance(self.model, torch.nn.Module):
                # PyTorch model
                input_tensor = input_tensor.to(self.device)
                logits = self.model(input_tensor)
                probabilities = F.softmax(logits, dim=1)
                probabilities = probabilities.cpu().numpy()[0]
            else:
                # ONNX model
                input_name = self.model.get_inputs()[0].name
                logits = self.model.run(None, {input_name: input_tensor.numpy()})[0]
                probabilities = self._softmax(logits)[0]
        
        # Get predictions
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])
        
        inference_time = (time.time() - start_time) * 1000  # milliseconds
        
        result = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "inference_time_ms": inference_time
        }
        
        if return_probabilities:
            result["probabilities"] = probabilities.tolist()
        
        return result
    
    @staticmethod
    def _softmax(x):
        """Softmax function for ONNX outputs."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Global inference engine
_inference_engine = None


def load_model(model_path: str):
    """Load model for inference."""
    global _inference_engine
    _inference_engine = ModelInference(model_path)
    _inference_engine.load_model()
    return _inference_engine


async def predict_wafer(
    wafer_map: np.ndarray,
    wafer_id: str,
    include_gradcam: bool = False,
    gradcam_layer: str = "layer4"
) -> Dict:
    """
    Predict yield for a wafer map.
    
    Args:
        wafer_map: Wafer map image array
        wafer_id: Wafer identifier
        include_gradcam: Generate Grad-CAM visualization
        gradcam_layer: Layer for Grad-CAM
        
    Returns:
        Prediction results
    """
    if _inference_engine is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    # Run inference
    result = _inference_engine.predict(wafer_map)
    
    # Map class index to defect class name
    defect_classes = [
        'Normal', 'EdgeEffect', 'CenterCluster', 'RingPattern',
        'QuadrantFailure', 'Scratch', 'RandomFailure', 'MixedMode'
    ]
    
    predicted_class_name = defect_classes[result['predicted_class']]
    
    # Build response
    response = {
        "wafer_id": wafer_id,
        "prediction": {
            "yield": None,  # STUB: yield regression not implemented — needs a separate model head
            "defect_class": predicted_class_name,
            "defect_probabilities": {
                defect_classes[i]: float(result['probabilities'][i])
                for i in range(len(defect_classes))
            },
            "confidence": result['confidence'],
            "uncertainty": 1.0 - result['confidence']
        },
        "model_version": "placeholder",  # No trained model artifact exists yet
        "inference_time_ms": result['inference_time_ms'],
        "timestamp": datetime.now().isoformat()
    }
    
    # Add Grad-CAM if requested
    if include_gradcam:
        # STUB: Grad-CAM generation is not implemented; no URL to return
        response["grad_cam_url"] = None
    
    return response


async def batch_predict(
    job_id: str,
    wafer_ids: List[str],
    include_gradcam: bool = False,
    model_version: Optional[str] = None
):
    """
    Run batch prediction job.
    
    Args:
        job_id: Job identifier
        wafer_ids: List of wafer IDs to process
        include_gradcam: Generate Grad-CAM visualizations
        model_version: Model version to use
    """
    logger.info(f"Starting batch job {job_id} with {len(wafer_ids)} wafers")
    
    results = []
    for wafer_id in wafer_ids:
        try:
            # STUB: actual wafer map loading from storage is not implemented
            wafer_map = np.zeros((300, 300, 3), dtype=np.uint8)
            
            # Predict
            result = await predict_wafer(
                wafer_map,
                wafer_id,
                include_gradcam=include_gradcam
            )
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to predict wafer {wafer_id}: {e}")
            results.append({"wafer_id": wafer_id, "error": str(e)})
    
    # Store results (placeholder - would save to database/cache)
    logger.info(f"Batch job {job_id} completed with {len(results)} results")
    
    return results
