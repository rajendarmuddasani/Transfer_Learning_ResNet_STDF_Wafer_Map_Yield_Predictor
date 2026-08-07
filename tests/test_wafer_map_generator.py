"""Tests for WaferMapGenerator image generation pipeline."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.data.wafer_map_generator import WaferMapGenerator
from PIL import Image


def test_generator_default_params():
    gen = WaferMapGenerator()
    assert gen.image_size == 300
    assert gen.die_size == 3


def test_generator_custom_params():
    gen = WaferMapGenerator(image_size=64, die_size=4)
    assert gen.image_size == 64
    assert gen.die_size == 4


def test_generate_returns_uint8_array(grid_wafer):
    gen = WaferMapGenerator(image_size=64, die_size=4)
    img = gen.generate(grid_wafer)
    assert img.dtype == np.uint8


def test_generate_output_shape(grid_wafer):
    gen = WaferMapGenerator(image_size=64, die_size=4)
    img = gen.generate(grid_wafer)
    assert img.shape == (64, 64, 3)


def test_generate_not_all_black(grid_wafer):
    """Image must contain colored pixels (dies were drawn)."""
    gen = WaferMapGenerator(image_size=128, die_size=5)
    img = gen.generate(grid_wafer)
    assert img.max() > 0


def test_generate_has_green_pass_pixels(grid_wafer):
    """Bin1 (pass) dies should produce green-dominant pixels."""
    gen = WaferMapGenerator(image_size=128, die_size=8)
    img = gen.generate(grid_wafer)
    # Green channel should have nonzero values
    assert img[:, :, 1].max() > 0


def test_generate_large_wafer(large_wafer):
    gen = WaferMapGenerator(image_size=64, die_size=3)
    img = gen.generate(large_wafer)
    assert img.shape == (64, 64, 3)
    assert img.dtype == np.uint8


def test_color_map_has_pass():
    gen = WaferMapGenerator()
    assert 'PASS' in gen.COLORS
    assert gen.COLORS['PASS'] == (0, 255, 0)


def test_color_map_has_fail_bins():
    gen = WaferMapGenerator()
    for bin_key in ['FAIL_BIN2', 'FAIL_BIN3', 'FAIL_BIN4']:
        assert bin_key in gen.COLORS


def test_coordinate_normalization_shape(grid_wafer):
    gen = WaferMapGenerator(image_size=64, die_size=4)
    norm = gen._normalize_coordinates(grid_wafer.coordinates, 64, 4)
    assert norm.shape == grid_wafer.coordinates.shape


def test_coordinate_normalization_within_bounds(grid_wafer):
    gen = WaferMapGenerator(image_size=64, die_size=4)
    norm = gen._normalize_coordinates(grid_wafer.coordinates, 64, 4)
    assert norm.min() >= 0
    assert norm.max() < 64

def test_draw_die_uses_exact_odd_size():
    generator = WaferMapGenerator(image_size=16, die_size=3)
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    generator._draw_die(image, 8, 8, (255, 0, 0))

    colored = np.all(image == np.array([255, 0, 0], dtype=np.uint8), axis=2)
    assert int(colored.sum()) == 9


def test_saved_image_preserves_rgb_channels(tmp_path):
    generator = WaferMapGenerator(image_size=4, die_size=1)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[1, 1] = (255, 0, 0)
    path = tmp_path / "rgb.png"

    generator._save_image(image, str(path))

    saved = np.asarray(Image.open(path).convert("RGB"))
    assert tuple(saved[1, 1]) == (255, 0, 0)
