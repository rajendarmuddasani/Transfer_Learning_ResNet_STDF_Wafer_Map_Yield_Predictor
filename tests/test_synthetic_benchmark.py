"""Contracts for the grouped synthetic wafer benchmark."""

import numpy as np

from src.contracts import DEFECT_CLASSES, IMAGE_SIZE
from src.data.synthetic_benchmark import (
    SampleSpec,
    build_sample_specs,
    generate_wafer_image,
    manifest_sha256,
)


def test_manifest_is_deterministic_and_split_groups_are_disjoint():
    first = build_sample_specs(variants_per_family=2)
    second = build_sample_specs(variants_per_family=2)

    assert first == second
    assert manifest_sha256(first) == manifest_sha256(second)
    split_groups = {
        split: {spec.family_id for spec in first if spec.split == split}
        for split in ("train", "validation", "confirmation")
    }
    assert split_groups["train"].isdisjoint(split_groups["validation"])
    assert split_groups["train"].isdisjoint(split_groups["confirmation"])
    assert split_groups["validation"].isdisjoint(split_groups["confirmation"])


def test_each_split_contains_every_class():
    specs = build_sample_specs(variants_per_family=1)
    for split in ("train", "validation", "confirmation"):
        assert {spec.class_name for spec in specs if spec.split == split} == set(
            DEFECT_CLASSES
        )


def test_image_generation_is_byte_reproducible():
    spec = build_sample_specs(variants_per_family=1)[0]

    first = generate_wafer_image(spec)
    second = generate_wafer_image(spec)

    assert first.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)


def test_unknown_class_is_rejected():
    spec = SampleSpec(
        split="train",
        class_name="Unknown",
        family_id="Unknown:1",
        family_seed=1,
        sample_seed=2,
        variant=0,
    )

    try:
        generate_wafer_image(spec)
    except ValueError as error:
        assert "Unknown defect class" in str(error)
    else:
        raise AssertionError("Unknown labels must not map to Normal")
