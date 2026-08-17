"""SSIM metric, configured for grayscale full-resolution images
(Architecture.md §7: metrics computed on full-resolution restored
images, not zoomed crops). Prefers scikit-image's reference
implementation; falls back to a NumPy implementation if unavailable so
the metrics module has no hard dependency."""
from __future__ import annotations

import numpy as np

try:
    from skimage.metrics import structural_similarity as _sk_ssim

    _HAS_SKIMAGE = True
except ImportError:  # pragma: no cover
    _HAS_SKIMAGE = False


def _ssim_numpy(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Minimal single-scale SSIM fallback (11x11 Gaussian window)."""
    from scipy.ndimage import gaussian_filter

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    mu_p = gaussian_filter(pred, sigma=1.5)
    mu_t = gaussian_filter(target, sigma=1.5)
    mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p * mu_t

    sigma_p2 = gaussian_filter(pred**2, sigma=1.5) - mu_p2
    sigma_t2 = gaussian_filter(target**2, sigma=1.5) - mu_t2
    sigma_pt = gaussian_filter(pred * target, sigma=1.5) - mu_pt

    num = (2 * mu_pt + c1) * (2 * sigma_pt + c2)
    den = (mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2)
    return float(np.mean(num / den))


def ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    pred = np.clip(pred, 0.0, data_range)
    target = np.clip(target, 0.0, data_range)
    if _HAS_SKIMAGE:
        return float(_sk_ssim(target, pred, data_range=data_range))
    return _ssim_numpy(pred, target, data_range)
