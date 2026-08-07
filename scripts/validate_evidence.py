"""Validate Project 4 claims, lineage, artifacts, and optional recomputation."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_public_evidence import build_evidence, file_sha256  # noqa: E402
from src.api.inference import verify_public_artifact  # noqa: E402
from src.contracts import PUBLIC_PROJECT_NAME  # noqa: E402


def load_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def assert_close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def validate_static() -> dict:
    claims = load_json("evidence/claims.json")
    schema = load_json("evidence/claims.schema.json")
    evaluation = load_json("evidence/public_synthetic_evaluation.json")
    manifest = load_json("evidence/public_synthetic_dataset_manifest.json")
    metadata = load_json("models/public_synthetic_resnet18_v1.json")
    Draft202012Validator(schema).validate(claims)

    verified = verify_public_artifact()
    assert claims["project"] == PUBLIC_PROJECT_NAME
    assert evaluation["project"] == PUBLIC_PROJECT_NAME
    assert claims["public_champion"] == metadata["model_version"]
    assert verified["model_sha256"] == evaluation["artifacts"]["onnx_sha256"]
    assert file_sha256(ROOT / evaluation["artifacts"]["metadata"]) == evaluation[
        "artifacts"
    ]["metadata_sha256"]
    assert evaluation["passes_all_gates"] is True
    assert evaluation["confirmation_opened_after_selection_freeze"] is True
    assert set(evaluation["group_overlap"].values()) == {0}
    assert manifest["generator_version"] == "synthetic_wafer_groups_v1"
    assert len(manifest["records"]) == 3_200

    metrics = evaluation["confirmation_metrics"]
    claim_values = {claim["claim"]: claim.get("value") for claim in claims["claims"]}
    mappings = {
        "Grouped synthetic confirmation accuracy": metrics["accuracy"],
        "Grouped synthetic confirmation macro F1": metrics["macro_f1"],
        "Grouped synthetic confirmation MCC": metrics["mcc"],
        "Minimum synthetic confirmation class recall": metrics[
            "minimum_class_recall"
        ],
        "Top-label expected calibration error": metrics["top_label_ece"],
        "Local ONNX CPU latency p95": evaluation["local_onnx_latency"]["p95_ms"],
    }
    for claim_name, expected in mappings.items():
        assert_close(claim_values[claim_name], expected, claim_name)

    for claim in claims["claims"]:
        if claim["safe"]:
            assert claim["allowed_wording"]
        if claim["evidence_class"] in {"unsupported", "target"}:
            assert claim["safe"] is False
            assert claim["allowed_wording"] == ""
        assert (ROOT / claim["source"]).exists(), claim["source"]

    return {"evaluation": evaluation}


def validate_recomputation(canonical: dict) -> None:
    with tempfile.TemporaryDirectory(prefix="project4-recompute-") as directory:
        temp_root = Path(directory)
        recomputed = build_evidence(
            variants_per_family=10,
            probe=False,
            model_path=temp_root / "model.onnx",
            metadata_path=temp_root / "model.json",
        )
    assert recomputed["selected_c"] == canonical["evaluation"]["selected_c"]
    assert_close(
        recomputed["temperature"],
        canonical["evaluation"]["temperature"],
        "temperature",
    )
    assert recomputed["validation_metrics"] == canonical["evaluation"][
        "validation_metrics"
    ]
    assert recomputed["confirmation_metrics"] == canonical["evaluation"][
        "confirmation_metrics"
    ]
    assert recomputed["gate_results"] == canonical["evaluation"]["gate_results"]
    print("OK: exact grouped synthetic selection and confirmation recomputation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true")
    arguments = parser.parse_args()

    canonical = validate_static()
    print("OK: claim schema, metric identity, split isolation, and artifact hashes")
    if arguments.recompute:
        validate_recomputation(canonical)
    print("Project 4 evidence validation passed")


if __name__ == "__main__":
    main()
