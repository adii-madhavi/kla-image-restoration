"""Paired degraded/ground-truth dataset.

Owns pairing and tensor conversion for training/validation. Validates
the input/GT relationship (matching stem filenames, exact 2x scale
relationship, single channel) rather than silently repairing mismatches
(Architecture.md §"src/data/dataset.py").
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.io import list_images, read_image


class PairingError(ValueError):
    """Raised when a degraded/GT pair fails a validation check. Never
    silently repaired — surfaced loudly per Project.md/Architecture.md."""


@dataclass
class PreprocessPolicy:
    """Explicit, named preprocessing policy — never an implicit default.

    clip_input: whether the degraded (NoisyLR) input is clipped to
        [0,1] before being fed to the model. Project.md §4 requires this
        decision to be *validated experimentally*, not assumed; the
        default here is False (preserve the out-of-range signal) and the
        choice actually used for the final model must be recorded in
        docs/experiment_log.md.
    """

    clip_input: bool = False
    clip_min: float = -1.0
    clip_max: float = 2.0  # generous safety bound, NOT [0,1] by default


class PairedRestorationDataset(Dataset):
    """Pairs a directory of degraded images with a directory of ground
    truth images by matching filename stem.

    Args:
        degraded_dir: directory of NoisyLR images.
        gt_dir: directory of clean GT images.
        scale: expected GT_size / degraded_size ratio (2 for the
            documented 128->256 and 256->512 cases).
        preprocess: PreprocessPolicy controlling degraded-value handling.
        augment: optional callable(degraded, gt) -> (degraded, gt) for
            paired-safe augmentation (src/data/augment.py).
        allowed_hw: optional set of allowed (H, W) degraded shapes; if
            given, unexpected shapes raise instead of being silently
            resized.
    """

    def __init__(
        self,
        degraded_dir: str | Path,
        gt_dir: str | Path,
        scale: int = 2,
        preprocess: Optional[PreprocessPolicy] = None,
        augment: Optional[Callable] = None,
        allowed_hw: Optional[set[tuple[int, int]]] = None,
    ):
        self.degraded_dir = Path(degraded_dir)
        self.gt_dir = Path(gt_dir)
        self.scale = scale
        self.preprocess = preprocess or PreprocessPolicy()
        self.augment = augment
        self.allowed_hw = allowed_hw

        self.pairs = self._build_pairs()
        if not self.pairs:
            raise PairingError(
                f"No matching degraded/GT pairs found between {self.degraded_dir} and {self.gt_dir}."
            )

    def _build_pairs(self) -> list[tuple[Path, Path]]:
        degraded_files = {p.stem: p for p in list_images(self.degraded_dir)}
        gt_files = {p.stem: p for p in list_images(self.gt_dir)}

        common = sorted(set(degraded_files) & set(gt_files))
        missing_gt = sorted(set(degraded_files) - set(gt_files))
        missing_deg = sorted(set(gt_files) - set(degraded_files))
        if missing_gt:
            raise PairingError(
                f"{len(missing_gt)} degraded image(s) have no matching GT, e.g. {missing_gt[:5]}"
            )
        if missing_deg:
            raise PairingError(
                f"{len(missing_deg)} GT image(s) have no matching degraded input, e.g. {missing_deg[:5]}"
            )
        return [(degraded_files[stem], gt_files[stem]) for stem in common]

    def __len__(self) -> int:
        return len(self.pairs)

    def _validate_shapes(self, degraded: np.ndarray, gt: np.ndarray, stem: str) -> None:
        dh, dw = degraded.shape
        gh, gw = gt.shape
        if self.allowed_hw is not None and (dh, dw) not in self.allowed_hw:
            raise PairingError(f"Unexpected degraded shape {(dh, dw)} for {stem}")
        if (gh, gw) != (dh * self.scale, dw * self.scale):
            raise PairingError(
                f"Pair '{stem}' does not respect the {self.scale}x scale contract: "
                f"degraded={dh}x{dw}, gt={gh}x{gw} (expected gt={dh*self.scale}x{dw*self.scale})"
            )

    def __getitem__(self, idx: int):
        deg_path, gt_path = self.pairs[idx]
        degraded = read_image(deg_path)
        gt = read_image(gt_path)

        self._validate_shapes(degraded, gt, deg_path.stem)

        if self.preprocess.clip_input:
            degraded = np.clip(degraded, self.preprocess.clip_min, self.preprocess.clip_max)
        gt = np.clip(gt, 0.0, 1.0)

        if self.augment is not None:
            degraded, gt = self.augment(degraded, gt)

        degraded_t = torch.from_numpy(np.ascontiguousarray(degraded)).float().unsqueeze(0)
        gt_t = torch.from_numpy(np.ascontiguousarray(gt)).float().unsqueeze(0)
        return {"degraded": degraded_t, "gt": gt_t, "name": deg_path.stem}


class InferenceImageDataset(Dataset):
    """Unpaired dataset for the standalone inference/evaluation path:
    just discovers images in `input_dir`, no GT required."""

    def __init__(self, input_dir: str | Path, preprocess: Optional[PreprocessPolicy] = None):
        self.input_dir = Path(input_dir)
        self.files = list_images(self.input_dir)
        if not self.files:
            raise PairingError(f"No supported images found in {self.input_dir}")
        self.preprocess = preprocess or PreprocessPolicy()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        arr = read_image(path)
        if self.preprocess.clip_input:
            arr = np.clip(arr, self.preprocess.clip_min, self.preprocess.clip_max)
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0)
        return {"degraded": tensor, "name": path.stem, "suffix": path.suffix}
