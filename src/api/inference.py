"""Hash-verified ONNX inference for the confirmed public classifier."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.contracts import (
    DEFECT_CLASSES,
    IMAGE_MEAN,
    IMAGE_SIZE,
    IMAGE_STD,
    PUBLIC_MODEL_METADATA_PATH,
    PUBLIC_MODEL_PATH,
    PUBLIC_MODEL_VERSION,
    ROOT,
)


EVALUATION_PATH = ROOT / "evidence" / "public_synthetic_evaluation.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_public_artifact(
    model_path: Path = PUBLIC_MODEL_PATH,
    metadata_path: Path = PUBLIC_MODEL_METADATA_PATH,
    evaluation_path: Path = EVALUATION_PATH,
) -> dict:
    for path in (model_path, metadata_path, evaluation_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    model_hash = file_sha256(model_path)
    metadata_hash = file_sha256(metadata_path)

    if metadata["model_version"] != PUBLIC_MODEL_VERSION:
        raise ValueError("public model version mismatch")
    if tuple(metadata["class_names"]) != DEFECT_CLASSES:
        raise ValueError("public model class order mismatch")
    if metadata["input"]["shape"] != [None, 3, IMAGE_SIZE, IMAGE_SIZE]:
        raise ValueError("public model input contract mismatch")
    if metadata["onnx_sha256"] != model_hash:
        raise ValueError("public ONNX SHA-256 mismatch")
    if evaluation["artifacts"]["onnx_sha256"] != model_hash:
        raise ValueError("evaluation ONNX SHA-256 mismatch")
    if evaluation["artifacts"]["metadata_sha256"] != metadata_hash:
        raise ValueError("evaluation metadata SHA-256 mismatch")
    if metadata["dataset_manifest_sha256"] != evaluation["dataset"]["manifest_sha256"]:
        raise ValueError("dataset manifest identity mismatch")
    if not evaluation["passes_all_gates"]:
        raise ValueError("public model did not pass promotion gates")

    return {
        "metadata": metadata,
        "evaluation": evaluation,
        "model_path": model_path,
        "model_sha256": model_hash,
        "metadata_sha256": metadata_hash,
    }


class ModelInference:
    """Run the confirmed ONNX classifier using its versioned contract."""

    def __init__(self, model_path: Path = PUBLIC_MODEL_PATH):
        verified = verify_public_artifact(model_path=model_path)
        self.metadata = verified["metadata"]
        self.evaluation = verified["evaluation"]
        self.model_sha256 = verified["model_sha256"]
        self.model_path = model_path
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != ["batch", 3, IMAGE_SIZE, IMAGE_SIZE]:
            raise ValueError("ONNX input signature mismatch")
        if len(outputs) != 1 or outputs[0].shape != ["batch", len(DEFECT_CLASSES)]:
            raise ValueError("ONNX output signature mismatch")

    @staticmethod
    def preprocess(image: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        resized = image.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
        )
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = (array - np.asarray(IMAGE_MEAN, dtype=np.float32)) / np.asarray(
            IMAGE_STD, dtype=np.float32
        )
        return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def predict(self, image: Image.Image | np.ndarray) -> dict:
        started = time.perf_counter()
        input_array = self.preprocess(image)
        logits = self.session.run(None, {self.input_name: input_array})[0]
        probabilities = self._softmax(logits)[0]
        class_index = int(np.argmax(probabilities))
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        return {
            "class_index": class_index,
            "defect_class": DEFECT_CLASSES[class_index],
            "confidence": float(probabilities[class_index]),
            "probabilities": {
                class_name: float(probabilities[index])
                for index, class_name in enumerate(DEFECT_CLASSES)
            },
            "inference_time_ms": elapsed_ms,
            "model_version": PUBLIC_MODEL_VERSION,
            "model_sha256": self.model_sha256,
        }


_inference_engine: ModelInference | None = None


def load_model(model_path: str | Path = PUBLIC_MODEL_PATH) -> ModelInference:
    global _inference_engine
    _inference_engine = ModelInference(Path(model_path))
    return _inference_engine


def get_model() -> ModelInference:
    if _inference_engine is None:
        raise RuntimeError("Confirmed model is not loaded")
    return _inference_engine
