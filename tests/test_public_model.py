"""Identity and behavior contracts for the confirmed ONNX model."""

import json
from pathlib import Path

import numpy as np

from src.api.inference import ModelInference, verify_public_artifact
from src.contracts import DEFECT_CLASSES, PUBLIC_MODEL_VERSION
from src.data.synthetic_benchmark import build_sample_specs, generate_wafer_image


ROOT = Path(__file__).resolve().parents[1]


def test_public_artifact_hashes_and_contract_match_evidence():
    verified = verify_public_artifact()

    assert verified["metadata"]["model_version"] == PUBLIC_MODEL_VERSION
    assert tuple(verified["metadata"]["class_names"]) == DEFECT_CLASSES
    assert verified["evaluation"]["passes_all_gates"] is True
    assert verified["model_sha256"] == verified["evaluation"]["artifacts"]["onnx_sha256"]


def test_public_model_returns_normalized_canonical_probabilities():
    engine = ModelInference()
    confirmation_spec = next(
        spec
        for spec in build_sample_specs(variants_per_family=1)
        if spec.split == "confirmation"
    )

    result = engine.predict(generate_wafer_image(confirmation_spec))

    assert result["model_version"] == PUBLIC_MODEL_VERSION
    assert result["defect_class"] in DEFECT_CLASSES
    assert tuple(result["probabilities"]) == DEFECT_CLASSES
    assert np.isclose(sum(result["probabilities"].values()), 1.0)


def test_evaluation_records_disjoint_confirmation_and_full_metrics():
    evidence = json.loads(
        (ROOT / "evidence" / "public_synthetic_evaluation.json").read_text(
            encoding="utf-8"
        )
    )

    assert evidence["confirmation_opened_after_selection_freeze"] is True
    assert set(evidence["group_overlap"].values()) == {0}
    assert evidence["passes_all_gates"] is True
    assert evidence["confirmation_metrics"]["samples"] == 800
    assert evidence["confirmation_metrics"]["minimum_class_recall"] >= 0.70
