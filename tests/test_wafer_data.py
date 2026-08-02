"""Tests for WaferData dataclass and yield computation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from src.data import WaferData


def test_wafer_data_construction(grid_wafer):
    assert grid_wafer.wafer_id == "TEST-W001"
    assert grid_wafer.die_count == 9


def test_wafer_yield_ratio(grid_wafer):
    assert grid_wafer.pass_count / grid_wafer.die_count == pytest.approx(7/9)


def test_wafer_fail_count(grid_wafer):
    assert grid_wafer.fail_count == 2
    assert grid_wafer.pass_count + grid_wafer.fail_count == grid_wafer.die_count


def test_wafer_coordinates_shape(grid_wafer):
    assert grid_wafer.coordinates.shape == (9, 2)


def test_wafer_bins_length(grid_wafer):
    assert len(grid_wafer.bins) == grid_wafer.die_count


def test_wafer_bins_unique_values(grid_wafer):
    unique = set(grid_wafer.bins.tolist())
    assert unique == {1, 2}


def test_wafer_lot_id_stored(grid_wafer):
    assert grid_wafer.lot_id == "TEST-LOT"


def test_large_wafer_yield(large_wafer):
    assert large_wafer.pass_count / large_wafer.die_count == pytest.approx(0.8)


def test_large_wafer_bins_range(large_wafer):
    assert large_wafer.bins.min() == 1
    assert large_wafer.bins.max() == 5


def test_wafer_num_stored():
    w = WaferData(
        wafer_id="W10",
        lot_id="L1",
        wafer_num=10,
        die_count=4,
        pass_count=3,
        fail_count=1,
        coordinates=np.zeros((4, 2)),
        bins=np.array([1, 1, 1, 2]),
    )
    assert w.wafer_num == 10
