"""Fail-closed smoke tests for imports, configuration, and API construction."""

import numpy as np
from fastapi import FastAPI

from src.api.main import app
from src.data import WaferData
from src.utils import load_config


def test_required_modules_import():
    assert WaferData is not None
    assert load_config is not None
    assert FastAPI is not None


def test_wafer_data_smoke():
    wafer_data = WaferData(
        wafer_id="TEST-W001",
        lot_id="TEST-LOT",
        wafer_num=1,
        die_count=3,
        pass_count=2,
        fail_count=1,
        coordinates=np.array([[0, 0], [1, 1], [2, 2]]),
        bins=np.array([1, 1, 2]),
    )

    assert wafer_data.wafer_id == "TEST-W001"
    assert wafer_data.pass_count / wafer_data.die_count == 2 / 3


def test_configuration_smoke():
    config = load_config("config/api_config.yaml")

    assert config["api"]["version"] == "2.0.0"
    assert config["model"]["production_model_path"].endswith(
        "public_synthetic_resnet18_v1.onnx"
    )


def test_fastapi_routes_exist():
    paths = set(app.openapi()["paths"])

    assert "/api/v1/classify-image" in paths
    assert "/api/v1/readiness" in paths
    assert "/metrics" in paths
