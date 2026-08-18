"""Paired-safe augmentation.

Only geometric transforms that are exactly consistent between degraded
input and GT (so the pixel-correspondence needed for a restoration loss
is preserved) plus optional calibrated synthetic degradation
(src/data/synthetic_degradation.py) applied to already-clean crops.
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image


class PairedGeometricAugment:
    """Random horizontal flip, vertical flip and 90-degree rotation,
    applied identically to (degraded, gt)."""

    def __init__(self, hflip: bool = True, vflip: bool = True, rot90: bool = True, p: float = 0.5):
        self.hflip = hflip
        self.vflip = vflip
        self.rot90 = rot90
        self.p = p

    def __call__(self, degraded: np.ndarray, gt: np.ndarray):
        if self.hflip and random.random() < self.p:
            degraded = np.ascontiguousarray(degraded[:, ::-1])
            gt = np.ascontiguousarray(gt[:, ::-1])
        if self.vflip and random.random() < self.p:
            degraded = np.ascontiguousarray(degraded[::-1, :])
            gt = np.ascontiguousarray(gt[::-1, :])
        if self.rot90 and random.random() < self.p:
            k = random.choice([1, 2, 3])
            degraded = np.ascontiguousarray(np.rot90(degraded, k))
            gt = np.ascontiguousarray(np.rot90(gt, k))
        return degraded, gt


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, degraded: np.ndarray, gt: np.ndarray):
        for t in self.transforms:
            degraded, gt = t(degraded, gt)
        return degraded, gt


class ApplySyntheticDegradation:
    """Re-derive a synthetic degraded input from GT with some
    probability, to broaden coverage beyond the fixed real pairs.

    ``CalibratedSyntheticDegradation`` samples a random downsample
    factor (e.g. 1.5x-4x) for realism/coverage purposes, which does
    NOT generally equal the dataset's fixed ``degraded_dir`` scale
    (exactly 2x, enforced by PairedRestorationDataset._validate_shapes
    and by the model's fixed upsampling factor). Feeding a
    variably-sized replacement into a batch alongside fixed-size real
    samples breaks DataLoader collation. So after generating the
    synthetic image, resample it (bicubic) to match the original
    degraded shape exactly - preserving the noise realism while
    respecting the hard scale contract.

    Defined at module level (not as a local closure) so it can be
    pickled by multiprocessing DataLoader workers on platforms using
    the 'spawn' start method (e.g. Windows)."""

    def __init__(self, synth, prob: float = 0.3):
        self.synth = synth
        self.prob = prob

    def __call__(self, degraded: np.ndarray, gt: np.ndarray):
        if random.random() < self.prob:
            target_shape = degraded.shape
            synthetic = self.synth(gt)
            if synthetic.shape != target_shape:
                img = Image.fromarray(synthetic.astype(np.float32), mode="F")
                synthetic = np.array(
                    img.resize((target_shape[1], target_shape[0]), resample=Image.BICUBIC),
                    dtype=np.float32,
                )
            degraded = synthetic
        return degraded, gt


def build_augment(cfg: dict | None):
    """Build the augmentation pipeline from a config dict, e.g.:

        augmentation:
          geometric: true
          synthetic_degradation: false
    """
    if not cfg or not cfg.get("geometric", True):
        return None
    transforms = [PairedGeometricAugment()]

    if cfg.get("synthetic_degradation", False):
        from src.data.synthetic_degradation import CalibratedSyntheticDegradation

        synth = CalibratedSyntheticDegradation(**cfg.get("synthetic_degradation_kwargs", {}))
        transforms.append(ApplySyntheticDegradation(synth, prob=cfg.get("synthetic_prob", 0.3)))

    return Compose(transforms)
