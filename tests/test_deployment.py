"""Static deployment and CI contracts for the confirmed public runtime."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_model_is_explicitly_tracked_and_canonical_binary():
    ignore_rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    model_path = ROOT / "models" / "onnx" / "public_synthetic_resnet18_v1.onnx"

    assert model_path.is_file()
    assert model_path.stat().st_size < 100 * 1024 * 1024
    assert "!models/onnx/public_synthetic_resnet18_v1.onnx" in ignore_rules
    assert "*.onnx binary" in attributes


def test_runtime_requirements_exclude_training_and_unwired_services():
    requirements = (ROOT / "requirements-public.txt").read_text(encoding="utf-8")

    assert "onnxruntime==" in requirements
    assert "fastapi==" in requirements
    assert "torch==" not in requirements
    assert "torchvision==" not in requirements
    assert "mlflow" not in requirements
    assert "sqlalchemy" not in requirements
    assert "redis" not in requirements


def test_browser_shell_uses_the_confirmed_public_product_name():
    index = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert "<title>Wafer Pattern Control Room</title>" in index
    assert "STDF Wafer Map Yield Predictor" not in index


def test_api_container_is_non_root_and_packages_confirmed_artifacts():
    dockerfile = (ROOT / "docker" / "Dockerfile.api").read_text(encoding="utf-8")

    assert "cgr.dev/chainguard/python:latest-dev AS builder" in dockerfile
    assert "cgr.dev/chainguard/python:latest" in dockerfile
    assert "requirements-public.txt" in dockerfile
    assert "public_synthetic_resnet18_v1.onnx" in dockerfile
    assert "public_synthetic_evaluation.json" in dockerfile
    assert "requirements-evidence.txt" not in dockerfile
    assert "ENTRYPOINT" in dockerfile
    assert "/api/v1/readiness" in dockerfile


def test_compose_contains_only_wired_services_and_requires_api_key():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "api:" in compose
    assert "frontend:" in compose
    assert "WAFER_CLASSIFIER_API_KEY: ${WAFER_CLASSIFIER_API_KEY:?" in compose
    for unsupported in ("postgres:", "redis:", "minio:", "mlflow:", "prometheus:"):
        assert unsupported not in compose


def test_ci_is_fail_closed_and_covers_all_release_surfaces():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert '["3.11", "3.12"]' in workflow
    assert "scripts/validate_evidence.py --recompute" in workflow
    assert "npm run lint" in workflow
    assert "npm run build" in workflow
    assert "npm audit --audit-level=high" in workflow
    assert "Authenticated runtime smoke" in workflow
    assert "actions/upload-artifact@v7" in workflow
    assert "|| true" not in workflow
    assert "Lint skipped" not in workflow
