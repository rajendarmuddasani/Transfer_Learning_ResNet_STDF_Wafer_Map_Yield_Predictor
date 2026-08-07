"""Strict class-contract tests for image datasets."""

from PIL import Image
import pytest

from src.contracts import DEFECT_CLASSES
from src.models.dataset import WaferMapDataset


def test_dataset_uses_canonical_class_order(tmp_path):
    class_dir = tmp_path / DEFECT_CLASSES[0]
    class_dir.mkdir()
    Image.new("RGB", (16, 16)).save(class_dir / "sample.png")

    dataset = WaferMapDataset(str(tmp_path))

    assert tuple(dataset.DEFECT_CLASSES) == DEFECT_CLASSES
    assert dataset[0][1] == 0


def test_dataset_rejects_unknown_flat_filename(tmp_path):
    Image.new("RGB", (16, 16)).save(tmp_path / "mystery.png")
    dataset = WaferMapDataset(str(tmp_path))

    with pytest.raises(ValueError, match="Unable to infer class label"):
        dataset[0]
