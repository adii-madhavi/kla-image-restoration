"""Tests for src/data/dataset.py pairing/validation logic — mismatches
must be surfaced loudly, never silently repaired (Architecture.md
§"src/data/dataset.py")."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.data.dataset import PairedRestorationDataset, PairingError


def test_valid_pairs_load(tiny_paired_dataset):
    ds = PairedRestorationDataset(
        degraded_dir=tiny_paired_dataset["degraded_dir"],
        gt_dir=tiny_paired_dataset["gt_dir"],
        scale=2,
    )
    assert len(ds) == 3
    sample = ds[0]
    assert sample["degraded"].shape == (1, 16, 16)
    assert sample["gt"].shape == (1, 32, 32)


def test_missing_gt_raises(tiny_paired_dataset):
    extra = tiny_paired_dataset["degraded_dir"] / "orphan.png"
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(extra)

    with pytest.raises(PairingError):
        PairedRestorationDataset(
            degraded_dir=tiny_paired_dataset["degraded_dir"],
            gt_dir=tiny_paired_dataset["gt_dir"],
            scale=2,
        )


def test_wrong_scale_raises_at_getitem(tiny_paired_dataset):
    ds = PairedRestorationDataset(
        degraded_dir=tiny_paired_dataset["degraded_dir"],
        gt_dir=tiny_paired_dataset["gt_dir"],
        scale=4,  # wrong on purpose — real relationship is 2x
    )
    with pytest.raises(PairingError):
        ds[0]


def test_no_pairs_found_raises(tmp_path):
    empty_a = tmp_path / "a"
    empty_b = tmp_path / "b"
    empty_a.mkdir()
    empty_b.mkdir()
    with pytest.raises(PairingError):
        PairedRestorationDataset(degraded_dir=empty_a, gt_dir=empty_b)
