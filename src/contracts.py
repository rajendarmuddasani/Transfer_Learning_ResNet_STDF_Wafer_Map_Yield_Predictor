"""Canonical public task and preprocessing contract."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PROJECT_NAME = "ResNet Wafer Pattern Classifier"
PUBLIC_API_NAME = "Wafer Pattern Classifier API"
PUBLIC_API_VERSION = "2.0.0"
PUBLIC_MODEL_VERSION = "public_synthetic_resnet18_v1"
PUBLIC_MODEL_PATH = ROOT / "models" / "onnx" / f"{PUBLIC_MODEL_VERSION}.onnx"
PUBLIC_MODEL_METADATA_PATH = ROOT / "models" / f"{PUBLIC_MODEL_VERSION}.json"

DEFECT_CLASSES = (
    "Normal",
    "EdgeEffect",
    "CenterCluster",
    "RingPattern",
    "QuadrantFailure",
    "Scratch",
    "RandomFailure",
    "MixedMode",
)

IMAGE_SIZE = 96
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
GENERATOR_VERSION = "synthetic_wafer_groups_v1"
