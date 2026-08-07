"""Deterministic grouped synthetic benchmark for wafer-pattern classification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from src.contracts import DEFECT_CLASSES, GENERATOR_VERSION, IMAGE_SIZE


@dataclass(frozen=True)
class SampleSpec:
    split: str
    class_name: str
    family_id: str
    family_seed: int
    sample_seed: int
    variant: int

    @property
    def sample_id(self) -> str:
        return f"{self.split}:{self.class_name}:{self.family_seed}:{self.variant}"


SPLIT_FAMILIES = {
    "train": range(0, 24),
    "validation": range(100, 106),
    "confirmation": range(200, 210),
}


def build_sample_specs(variants_per_family: int = 10) -> list[SampleSpec]:
    specs: list[SampleSpec] = []
    for class_index, class_name in enumerate(DEFECT_CLASSES):
        class_seed = (class_index + 1) * 10_000
        for split, family_offsets in SPLIT_FAMILIES.items():
            for family_offset in family_offsets:
                family_seed = class_seed + family_offset
                family_id = f"{class_name}:{family_seed}"
                for variant in range(variants_per_family):
                    specs.append(
                        SampleSpec(
                            split=split,
                            class_name=class_name,
                            family_id=family_id,
                            family_seed=family_seed,
                            sample_seed=family_seed * 100 + variant,
                            variant=variant,
                        )
                    )
    return specs


def manifest_payload(specs: Iterable[SampleSpec]) -> dict:
    records = [asdict(spec) | {"sample_id": spec.sample_id} for spec in specs]
    return {
        "generator_version": GENERATOR_VERSION,
        "class_names": list(DEFECT_CLASSES),
        "image_size": IMAGE_SIZE,
        "records": records,
    }


def manifest_sha256(specs: Iterable[SampleSpec]) -> str:
    canonical = json.dumps(
        manifest_payload(specs), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _family_parameters(class_name: str, family_seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(family_seed)
    return {
        "radius": float(rng.uniform(0.86, 0.96)),
        "center_x": float(rng.uniform(-0.12, 0.12)),
        "center_y": float(rng.uniform(-0.12, 0.12)),
        "ring_radius": float(rng.uniform(0.40, 0.62)),
        "ring_width": float(rng.uniform(0.055, 0.10)),
        "scratch_angle": float(rng.uniform(0.0, np.pi)),
        "scratch_offset": float(rng.uniform(-0.22, 0.22)),
        "quadrant": int(rng.integers(0, 4)),
        "edge_angle": float(rng.uniform(-np.pi, np.pi)),
        "edge_width": float(rng.uniform(0.7, 1.5)),
        "random_rate": float(rng.uniform(0.14, 0.23)),
        "class_name": class_name,
    }


def generate_wafer_image(spec: SampleSpec, image_size: int = IMAGE_SIZE) -> np.ndarray:
    if spec.class_name not in DEFECT_CLASSES:
        raise ValueError(f"Unknown defect class: {spec.class_name}")

    params = _family_parameters(spec.class_name, spec.family_seed)
    rng = np.random.default_rng(spec.sample_seed)
    grid_size = 30
    coordinates = np.linspace(-1.0, 1.0, grid_size)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    centered_x = xx - float(params["center_x"])
    centered_y = yy - float(params["center_y"])
    distance = np.sqrt(centered_x**2 + centered_y**2)
    wafer = distance <= float(params["radius"])

    base_rate = float(rng.uniform(0.008, 0.025))
    probability = np.full((grid_size, grid_size), base_rate, dtype=np.float64)
    class_name = spec.class_name

    if class_name == "EdgeEffect":
        angle = np.arctan2(centered_y, centered_x)
        angular_delta = np.angle(np.exp(1j * (angle - float(params["edge_angle"]))))
        region = (distance > float(params["radius"]) * 0.76) & (
            np.abs(angular_delta) < float(params["edge_width"])
        )
        probability[region] = rng.uniform(0.48, 0.78)
    elif class_name == "CenterCluster":
        region = distance < float(params["radius"]) * rng.uniform(0.22, 0.34)
        probability[region] = rng.uniform(0.48, 0.78)
    elif class_name == "RingPattern":
        region = np.abs(distance - float(params["ring_radius"])) < float(
            params["ring_width"]
        )
        probability[region] = rng.uniform(0.48, 0.76)
    elif class_name == "QuadrantFailure":
        quadrant = int(params["quadrant"])
        x_positive = centered_x > 0
        y_positive = centered_y > 0
        regions = (
            x_positive & y_positive,
            ~x_positive & y_positive,
            ~x_positive & ~y_positive,
            x_positive & ~y_positive,
        )
        probability[regions[quadrant]] = rng.uniform(0.38, 0.66)
    elif class_name == "Scratch":
        angle = float(params["scratch_angle"])
        line_distance = np.abs(
            centered_y * np.cos(angle)
            - centered_x * np.sin(angle)
            - float(params["scratch_offset"])
        )
        region = line_distance < rng.uniform(0.045, 0.085)
        probability[region] = rng.uniform(0.58, 0.86)
    elif class_name == "RandomFailure":
        probability[wafer] = float(params["random_rate"])
    elif class_name == "MixedMode":
        center_region = distance < float(params["radius"]) * 0.24
        edge_region = distance > float(params["radius"]) * 0.80
        probability[center_region | edge_region] = rng.uniform(0.36, 0.62)

    failed = (rng.random((grid_size, grid_size)) < probability) & wafer
    passed = wafer & ~failed

    canvas = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    canvas[passed] = np.array([34, 174, 91], dtype=np.uint8)
    canvas[failed] = np.array([218, 65, 54], dtype=np.uint8)
    image = Image.fromarray(canvas).resize((image_size, image_size), Image.Resampling.NEAREST)

    brightness = float(rng.uniform(0.88, 1.12))
    array = np.clip(np.asarray(image, dtype=np.float32) * brightness, 0, 255)
    noise = rng.normal(0.0, rng.uniform(0.0, 2.2), size=array.shape)
    array = np.clip(array + noise, 0, 255).astype(np.uint8)

    rendered = Image.fromarray(array)
    draw = ImageDraw.Draw(rendered)
    draw.ellipse(
        (1, 1, image_size - 2, image_size - 2),
        outline=(214, 222, 220),
        width=1,
    )
    return np.asarray(rendered, dtype=np.uint8)


class SyntheticWaferDataset(Dataset):
    def __init__(self, specs: Iterable[SampleSpec], transform=None):
        self.specs = list(specs)
        self.transform = transform
        self.class_to_idx = {name: index for index, name in enumerate(DEFECT_CLASSES)}

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, index: int):
        spec = self.specs[index]
        image = Image.fromarray(generate_wafer_image(spec))
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_idx[spec.class_name]
