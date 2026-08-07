"""Contracts for public WM-811K metric and artifact claims."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_wm811k_accuracy_is_ledgered_and_artifact_qualified():
    claims = json.loads((ROOT / "evidence" / "claims.json").read_text(encoding="utf-8"))
    accuracy_claims = [
        claim
        for claim in claims["claims"]
        if claim["claim"] == "WM-811K test accuracy"
    ]

    assert len(accuracy_claims) == 1
    claim = accuracy_claims[0]
    assert claim["value"] == 0.89
    assert claim["evidence_class"] in {"reproduced", "historical", "unsupported"}

    result_path = ROOT / "artifacts" / "wm811k_results.json"
    model_paths = (
        ROOT / "artifacts" / "resnet18_wm811k_best.pth",
        ROOT / "artifacts" / "resnet18_wm811k.onnx",
    )
    if not result_path.is_file() or not all(path.is_file() for path in model_paths):
        assert claim["evidence_class"] == "unsupported"


def test_readme_does_not_present_unverified_wm811k_outputs_as_validated():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    result_path = ROOT / "artifacts" / "wm811k_results.json"
    onnx_path = ROOT / "artifacts" / "resnet18_wm811k.onnx"

    if not result_path.is_file():
        assert "**Test Accuracy** | **89.0%**" not in readme
    if not onnx_path.is_file():
        assert "ONNX Export | Validated, production-ready" not in readme
