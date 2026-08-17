"""Tests for src/data/io.py: file loading and saving must not silently
clip or rescale values it shouldn't (Project.md §4)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.io import list_images, read_image, write_image


def test_list_images_sorted_and_filtered(tmp_path: Path):
    (tmp_path / "b.png").write_bytes(b"")
    (tmp_path / "a.png").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    files = list_images(tmp_path)
    names = [f.name for f in files]
    assert names == ["a.png", "b.png"]  # deterministic order, non-image excluded


def test_write_and_read_uint8_roundtrip(tmp_path: Path):
    arr = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    path = tmp_path / "out.png"
    write_image(path, arr, encoding="uint8")
    loaded = read_image(path)
    assert loaded.shape == (8, 8)
    # uint8 quantization introduces small error, but should be tiny
    assert np.abs(loaded - arr).max() < 0.01


def test_npy_preserves_out_of_range_values(tmp_path: Path):
    arr = np.array([[-0.3, 0.0], [1.0, 1.4]], dtype=np.float32)
    path = tmp_path / "noisy_lr.npy"
    np.save(path, arr)
    loaded = read_image(path)
    # values below 0 / above 1 must survive unchanged — NoisyLR is
    # explicitly allowed to extend outside [0,1] (Project.md §3)
    np.testing.assert_allclose(loaded, arr)


def test_write_float32_encoding_no_clip(tmp_path: Path):
    arr = np.array([[-0.5, 1.5]], dtype=np.float32)
    path = tmp_path / "out"
    write_image(path, arr, encoding="float32")
    loaded = np.load(path.with_suffix(".npy"))
    np.testing.assert_allclose(loaded, arr)
