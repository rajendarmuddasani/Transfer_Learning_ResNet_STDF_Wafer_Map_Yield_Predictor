"""Integration contracts for the confirmed wafer-pattern classifier API."""

import io

from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app
from src.contracts import (
    DEFECT_CLASSES,
    PUBLIC_API_NAME,
    PUBLIC_API_VERSION,
    PUBLIC_MODEL_VERSION,
)
from src.data.synthetic_benchmark import build_sample_specs, generate_wafer_image


def confirmation_png() -> bytes:
    spec = next(
        spec
        for spec in build_sample_specs(variants_per_family=1)
        if spec.split == "confirmation"
    )
    output = io.BytesIO()
    Image.fromarray(generate_wafer_image(spec)).save(output, format="PNG")
    return output.getvalue()


def test_readiness_binds_model_version_and_hash():
    with TestClient(app) as client:
        response = client.get("/api/v1/readiness")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["model_version"] == PUBLIC_MODEL_VERSION
    assert len(payload["model_sha256"]) == 64


def test_root_and_health_share_the_public_api_contract():
    with TestClient(app) as client:
        root = client.get("/")
        health = client.get("/api/v1/health")

    assert root.status_code == 200
    assert root.json()["name"] == PUBLIC_API_NAME
    assert root.json()["version"] == PUBLIC_API_VERSION
    assert health.status_code == 200
    assert health.json()["version"] == PUBLIC_API_VERSION
    assert app.title == PUBLIC_API_NAME
    assert app.version == PUBLIC_API_VERSION


def test_classify_image_returns_canonical_probabilities_and_request_id():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/classify-image",
            headers={"X-Request-ID": "project4-test"},
            files={"wafer_map_image": ("confirmation.png", confirmation_png(), "image/png")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_version"] == PUBLIC_MODEL_VERSION
    assert payload["defect_class"] in DEFECT_CLASSES
    assert tuple(payload["defect_probabilities"]) == DEFECT_CLASSES
    assert payload["request_id"] == "project4-test"
    assert response.headers["X-Request-ID"] == "project4-test"


def test_stdf_and_batch_paths_are_explicitly_unsupported():
    with TestClient(app) as client:
        stdf_response = client.post(
            "/api/v1/predict",
            files={"stdf_file": ("sample.stdf", b"not-a-real-stdf", "application/octet-stream")},
        )
        batch_response = client.post("/api/v1/predict/batch")

    assert stdf_response.status_code == 501
    assert "outside the confirmed" in stdf_response.json()["detail"]
    assert batch_response.status_code == 501


def test_invalid_or_oversized_image_is_rejected_before_inference(monkeypatch):
    monkeypatch.setattr("src.api.routes.predict.MAX_UPLOAD_BYTES", 5)
    with TestClient(app) as client:
        oversized = client.post(
            "/api/v1/classify-image",
            files={"wafer_map_image": ("large.png", b"123456", "image/png")},
        )
        unsupported = client.post(
            "/api/v1/classify-image",
            files={"wafer_map_image": ("wafer.txt", b"data", "text/plain")},
        )

    assert oversized.status_code == 413
    assert unsupported.status_code == 415


def test_production_requires_api_key(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("WAFER_CLASSIFIER_API_KEY", "test-only-key")
    files = {"wafer_map_image": ("confirmation.png", confirmation_png(), "image/png")}
    with TestClient(app) as client:
        unauthorized = client.post("/api/v1/classify-image", files=files)
        authorized = client.post(
            "/api/v1/classify-image",
            headers={"X-API-Key": "test-only-key"},
            files={"wafer_map_image": ("confirmation.png", confirmation_png(), "image/png")},
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_model_registry_and_metrics_report_real_runtime_state():
    with TestClient(app) as client:
        models = client.get("/api/v1/models")
        metrics = client.get("/metrics")
        promotion = client.post(f"/api/v1/models/{PUBLIC_MODEL_VERSION}/promote")

    assert models.status_code == 200
    assert models.json()[0]["stage"] == "CONFIRMED_SYNTHETIC"
    assert models.json()[0]["accuracy"] == 0.93625
    assert "wafer_classifier_requests_total" in metrics.text
    assert promotion.status_code == 501
