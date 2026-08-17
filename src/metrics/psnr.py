"""PSNR metric. Uses data_range=1.0 since prediction/GT are expected to
be in [0,1] at evaluation time (Architecture.md §7)."""
from __future__ import annotations

import numpy as np


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    pred = np.clip(pred, 0.0, data_range).astype(np.float64)
    target = np.clip(target, 0.0, data_range).astype(np.float64)
    mse = np.mean((pred - target) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10((data_range**2) / mse)
