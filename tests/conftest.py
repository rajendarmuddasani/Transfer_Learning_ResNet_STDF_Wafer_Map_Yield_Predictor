"""Shared fixtures for ResNet wafer map yield predictor tests."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from src.data import WaferData


@pytest.fixture
def grid_wafer():
    """3×3 grid wafer: 7 pass (bin1), 2 fail (bin2)."""
    coords = np.array([[x, y] for y in range(3) for x in range(3)])
    bins = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2])
    return WaferData(
        wafer_id="TEST-W001",
        lot_id="TEST-LOT",
        wafer_num=1,
        die_count=9,
        pass_count=7,
        fail_count=2,
        coordinates=coords,
        bins=bins,
    )


@pytest.fixture
def large_wafer():
    """5×5 grid wafer: 20 pass, 5 fail (bins 2-5)."""
    coords = np.array([[x, y] for y in range(5) for x in range(5)])
    bins = np.array([1] * 20 + [2, 3, 4, 5, 2])
    return WaferData(
        wafer_id="TEST-W002",
        lot_id="TEST-LOT-B",
        wafer_num=2,
        die_count=25,
        pass_count=20,
        fail_count=5,
        coordinates=coords,
        bins=bins,
    )
