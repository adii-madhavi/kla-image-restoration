"""Shared pytest fixtures: build tiny synthetic degraded/GT directories
so tests never depend on the (large, non-public) official dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(arr, 0.0, 1.0)
    Image.fromarray((clipped * 255).astype(np.uint8), mode="L").save(path)


@pytest.fixture
def tiny_paired_dataset(tmp_path: Path) -> dict:
    """3 pairs of 16x16 degraded / 32x32 GT images (scale=2)."""
    degraded_dir = tmp_path / "degraded"
    gt_dir = tmp_path / "gt"
    rng = np.random.default_rng(42)

    names = []
    for i in range(3):
        gt = rng.random((32, 32)).astype(np.float32)
        degraded = np.array(
            Image.fromarray((gt * 255).astype(np.uint8)).resize((16, 16), Image.BICUBIC)
        ).astype(np.float32) / 255.0
        name = f"img_{i:03d}"
        _write_png(degraded_dir / f"{name}.png", degraded)
        _write_png(gt_dir / f"{name}.png", gt)
        names.append(name)

    return {"degraded_dir": degraded_dir, "gt_dir": gt_dir, "names": names}


@pytest.fixture
def tiny_input_dir(tmp_path: Path) -> Path:
    input_dir = tmp_path / "input"
    rng = np.random.default_rng(7)
    for i in range(2):
        arr = rng.random((16, 16)).astype(np.float32)
        _write_png(input_dir / f"sample_{i:03d}.png", arr)
    return input_dir
