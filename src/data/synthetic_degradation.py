"""Synthetic degradation from clean GT images.

Only used AFTER the real paired data has been analyzed
(scripts/audit_dataset.py) — parameters are exposed explicitly so the
synthetic distribution can be calibrated against measured statistics of
the real NoisyLR data rather than guessed (Architecture.md
§"src/data/synthetic_degradation.py").

Implements exactly the three official benchmark degradation mechanisms
(Project.md §3 / Source_of_Truth.md §8):
    1. additive Gaussian noise      Y = X + N_G
    2. multiplicative speckle noise Y = X * N_S
    3. spatial downsampling

The order is randomized per call, since the official order is not
disclosed and must not be assumed fixed (Project.md §1, CLAUDE.md §2).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


def add_gaussian_noise(x: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(loc=0.0, scale=sigma, size=x.shape).astype(np.float32)
    return x + noise


def add_speckle_noise(x: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(loc=1.0, scale=sigma, size=x.shape).astype(np.float32)
    return x * noise


def downsample(x: np.ndarray, factor: float) -> np.ndarray:
    h, w = x.shape
    new_h, new_w = max(1, round(h / factor)), max(1, round(w / factor))
    img = Image.fromarray(np.clip(x, 0.0, 1.0).astype(np.float32), mode="F")
    resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
    return np.array(resized, dtype=np.float32)


@dataclass
class CalibratedSyntheticDegradation:
    """Applies a random subset/order of {Gaussian noise, speckle noise,
    downsampling} to a clean image. Ranges default to values consistent
    with Architecture.md's discussion (sigma increasing with severity,
    downsampling factor ~1.5x-4x) but MUST be recalibrated against the
    real dataset statistics from scripts/audit_dataset.py before being
    treated as final (see docs/experiment_log.md)."""

    gaussian_sigma_range: tuple[float, float] = (0.01, 0.08)
    speckle_sigma_range: tuple[float, float] = (0.02, 0.15)
    downsample_factor_range: tuple[float, float] = (1.5, 4.0)
    always_downsample: bool = True
    gaussian_prob: float = 0.7
    speckle_prob: float = 0.7

    def __call__(self, gt: np.ndarray) -> np.ndarray:
        ops = []
        if random.random() < self.gaussian_prob:
            sigma = random.uniform(*self.gaussian_sigma_range)
            ops.append(lambda a: add_gaussian_noise(a, sigma))
        if random.random() < self.speckle_prob:
            sigma = random.uniform(*self.speckle_sigma_range)
            ops.append(lambda a: add_speckle_noise(a, sigma))
        if self.always_downsample:
            factor = random.uniform(*self.downsample_factor_range)
            ops.append(lambda a: downsample(a, factor))

        random.shuffle(ops)  # order is not disclosed/fixed officially
        out = gt.astype(np.float32).copy()
        for op in ops:
            out = op(out)
        return out
