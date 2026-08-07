"""Build validation-selected ResNet-18 evidence on grouped synthetic wafers."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sklearn
import torch
import torch.nn as nn
import torchvision
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.contracts import (  # noqa: E402
    DEFECT_CLASSES,
    IMAGE_MEAN,
    IMAGE_SIZE,
    IMAGE_STD,
    PUBLIC_MODEL_METADATA_PATH,
    PUBLIC_MODEL_PATH,
    PUBLIC_MODEL_VERSION,
    PUBLIC_PROJECT_NAME,
)
from src.data.synthetic_benchmark import (  # noqa: E402
    SyntheticWaferDataset,
    build_sample_specs,
    manifest_payload,
    manifest_sha256,
)
from src.evaluation import fit_temperature, softmax, summarize_predictions  # noqa: E402
from src.models.resnet_model import ResNetTransferLearning  # noqa: E402

MANIFEST_PATH = ROOT / "evidence" / "public_synthetic_dataset_manifest.json"
EVALUATION_PATH = ROOT / "evidence" / "public_synthetic_evaluation.json"
CANDIDATE_C = (0.01, 0.1, 1.0, 10.0)
PROMOTION_GATES = {
    "minimum_accuracy": 0.90,
    "minimum_macro_f1": 0.88,
    "minimum_class_recall": 0.70,
    "maximum_top_label_ece": 0.10,
    "maximum_onnx_parity_error": 1e-4,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )


def extract_features(model, specs, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    dataset = SyntheticWaferDataset(specs, transform=build_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feature_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            feature_rows.append(model.extract_features(images).cpu().numpy())
            label_rows.append(labels.numpy())
    return np.concatenate(feature_rows), np.concatenate(label_rows)


def select_head(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
) -> tuple[LogisticRegression, list[dict]]:
    candidates: list[tuple[tuple[float, float, float], LogisticRegression, dict]] = []
    for c_value in CANDIDATE_C:
        classifier = LogisticRegression(
            C=c_value,
            max_iter=1_000,
            solver="lbfgs",
            random_state=42,
        )
        classifier.fit(train_features, train_labels)
        logits = classifier.decision_function(validation_features)
        metrics = summarize_predictions(
            validation_labels, softmax(logits), DEFECT_CLASSES
        )
        record = {"c": c_value, "metrics": metrics}
        score = (metrics["macro_f1"], metrics["accuracy"], -metrics["log_loss"])
        candidates.append((score, classifier, record))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], [candidate[2] for candidate in candidates]


class CalibratedClassifier(nn.Module):
    def __init__(self, model: ResNetTransferLearning, temperature: float):
        super().__init__()
        self.model = model
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images) / self.temperature


def install_head(model: ResNetTransferLearning, classifier: LogisticRegression) -> None:
    with torch.no_grad():
        model.model.fc.weight.copy_(torch.from_numpy(classifier.coef_).float())
        model.model.fc.bias.copy_(torch.from_numpy(classifier.intercept_).float())


def export_onnx(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def onnx_logits(path: Path, specs, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    dataset = SyntheticWaferDataset(specs, transform=build_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, batch_labels in loader:
        logits.append(session.run(None, {input_name: images.numpy()})[0])
        labels.append(batch_labels.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def benchmark_latency(path: Path, sample: np.ndarray, iterations: int = 100) -> dict:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    for _ in range(10):
        session.run(None, {input_name: sample})
    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        session.run(None, {input_name: sample})
        timings.append((time.perf_counter() - started) * 1_000.0)
    return {
        "iterations": iterations,
        "batch_size": int(sample.shape[0]),
        "mean_ms": float(np.mean(timings)),
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "p99_ms": float(np.percentile(timings, 99)),
        "scope": "local CPU ONNX Runtime; not a production SLO",
    }


def build_evidence(
    variants_per_family: int,
    probe: bool,
    model_path: Path = PUBLIC_MODEL_PATH,
    metadata_path: Path = PUBLIC_MODEL_METADATA_PATH,
) -> dict:
    torch.manual_seed(42)
    np.random.seed(42)
    specs = build_sample_specs(variants_per_family=variants_per_family)
    train_specs = [spec for spec in specs if spec.split == "train"]
    validation_specs = [spec for spec in specs if spec.split == "validation"]
    confirmation_specs = [spec for spec in specs if spec.split == "confirmation"]

    backbone = ResNetTransferLearning(
        architecture="resnet18",
        num_classes=len(DEFECT_CLASSES),
        pretrained=True,
        freeze_backbone=True,
    )
    train_features, train_labels = extract_features(backbone, train_specs)
    validation_features, validation_labels = extract_features(backbone, validation_specs)
    classifier, candidates = select_head(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    )
    validation_logits = classifier.decision_function(validation_features)
    temperature, _ = fit_temperature(validation_logits, validation_labels)
    validation_metrics = summarize_predictions(
        validation_labels,
        softmax(validation_logits, temperature),
        DEFECT_CLASSES,
    )
    selected_c = float(classifier.C)

    result = {
        "mode": "probe" if probe else "confirmation",
        "selected_c": selected_c,
        "temperature": temperature,
        "candidate_selection": candidates,
        "validation_metrics": validation_metrics,
        "dataset": {
            "variants_per_family": variants_per_family,
            "train_samples": len(train_specs),
            "validation_samples": len(validation_specs),
            "confirmation_samples": len(confirmation_specs),
            "manifest_sha256": manifest_sha256(specs),
        },
    }
    if probe:
        return result

    install_head(backbone, classifier)
    calibrated_model = CalibratedClassifier(backbone, temperature)
    export_onnx(calibrated_model, model_path)

    confirmation_logits, confirmation_labels = onnx_logits(
        model_path, confirmation_specs
    )
    confirmation_probabilities = softmax(confirmation_logits)
    confirmation_metrics = summarize_predictions(
        confirmation_labels,
        confirmation_probabilities,
        DEFECT_CLASSES,
        include_bootstrap=True,
    )

    parity_specs = confirmation_specs[:32]
    parity_dataset = SyntheticWaferDataset(parity_specs, transform=build_transform())
    parity_images = torch.stack([parity_dataset[index][0] for index in range(len(parity_dataset))])
    with torch.inference_mode():
        pytorch_output = calibrated_model(parity_images).numpy()
    parity_session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    onnx_output = parity_session.run(
        None, {parity_session.get_inputs()[0].name: parity_images.numpy()}
    )[0]
    maximum_parity_error = float(np.max(np.abs(pytorch_output - onnx_output)))

    gates = {
        "accuracy": confirmation_metrics["accuracy"]
        >= PROMOTION_GATES["minimum_accuracy"],
        "macro_f1": confirmation_metrics["macro_f1"]
        >= PROMOTION_GATES["minimum_macro_f1"],
        "minimum_class_recall": confirmation_metrics["minimum_class_recall"]
        >= PROMOTION_GATES["minimum_class_recall"],
        "top_label_ece": confirmation_metrics["top_label_ece"]
        <= PROMOTION_GATES["maximum_top_label_ece"],
        "onnx_parity": maximum_parity_error
        <= PROMOTION_GATES["maximum_onnx_parity_error"],
        "group_overlap": True,
    }
    model_hash = file_sha256(model_path)
    metadata = {
        "model_version": PUBLIC_MODEL_VERSION,
        "task": "synthetic wafer-pattern classification",
        "architecture": "ResNet-18 ImageNet frozen feature extractor with validation-selected linear head",
        "class_names": list(DEFECT_CLASSES),
        "input": {
            "shape": [None, 3, IMAGE_SIZE, IMAGE_SIZE],
            "color_order": "RGB",
            "mean": list(IMAGE_MEAN),
            "std": list(IMAGE_STD),
        },
        "calibration_temperature": temperature,
        "dataset_manifest_sha256": manifest_sha256(specs),
        "onnx_sha256": model_hash,
        "evidence": "evidence/public_synthetic_evaluation.json",
        "limitations": [
            "Validated only on independently generated synthetic wafer maps.",
            "Not evidence of WM-811K performance or production silicon behavior.",
            "STDF parsing is outside the confirmed image-classification boundary.",
        ],
    }
    write_json(metadata_path, metadata)

    split_groups = {
        split: {spec.family_id for spec in specs if spec.split == split}
        for split in ("train", "validation", "confirmation")
    }
    result.update(
        {
            "project": PUBLIC_PROJECT_NAME,
            "benchmark_version": "public_synthetic_wafer_v1",
            "evidence_class": "reproduced",
            "data_scope": "independently generated grouped synthetic wafer maps",
            "task": "eight-class wafer-pattern classification",
            "confirmation_opened_after_selection_freeze": True,
            "group_overlap": {
                "train_validation": len(split_groups["train"] & split_groups["validation"]),
                "train_confirmation": len(split_groups["train"] & split_groups["confirmation"]),
                "validation_confirmation": len(
                    split_groups["validation"] & split_groups["confirmation"]
                ),
            },
            "promotion_gates": PROMOTION_GATES,
            "gate_results": gates,
            "passes_all_gates": all(gates.values()),
            "confirmation_metrics": confirmation_metrics,
            "baselines": {
                "uniform_random_accuracy": 1.0 / len(DEFECT_CLASSES),
                "uniform_random_macro_f1": 1.0 / len(DEFECT_CLASSES),
            },
            "artifacts": {
                "onnx": str(model_path.relative_to(ROOT)).replace("\\", "/")
                if model_path.is_relative_to(ROOT)
                else str(model_path),
                "onnx_sha256": model_hash,
                "metadata": str(metadata_path.relative_to(ROOT)).replace("\\", "/")
                if metadata_path.is_relative_to(ROOT)
                else str(metadata_path),
                "metadata_sha256": file_sha256(metadata_path),
                "maximum_pytorch_onnx_logit_error": maximum_parity_error,
            },
            "local_onnx_latency": benchmark_latency(
                model_path, parity_images[:1].numpy()
            ),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "onnxruntime": ort.__version__,
                "scikit_learn": sklearn.__version__,
                "device": "cpu",
            },
            "limitations": metadata["limitations"],
        }
    )
    return result | {"manifest": manifest_payload(specs)}


def write_canonical(result: dict) -> None:
    manifest = result.pop("manifest")
    write_json(MANIFEST_PATH, manifest)
    write_json(EVALUATION_PATH, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--variants-per-family", type=int, default=10)
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    if arguments.write and arguments.probe:
        raise SystemExit("Probe mode cannot write canonical artifacts")

    result = build_evidence(arguments.variants_per_family, arguments.probe)
    summary = {
        "mode": result["mode"],
        "selected_c": result["selected_c"],
        "temperature": result["temperature"],
        "dataset": result["dataset"],
        "validation_metrics": result["validation_metrics"],
    }
    if not arguments.probe:
        summary["confirmation_metrics"] = result["confirmation_metrics"]
        summary["passes_all_gates"] = result["passes_all_gates"]
    print(json.dumps(summary, indent=2))
    if arguments.write:
        write_canonical(result)
        print(f"Wrote {EVALUATION_PATH.relative_to(ROOT)}")
        print(f"Wrote {MANIFEST_PATH.relative_to(ROOT)}")
        print(f"Wrote {PUBLIC_MODEL_PATH.relative_to(ROOT)}")
        print(f"Wrote {PUBLIC_MODEL_METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
