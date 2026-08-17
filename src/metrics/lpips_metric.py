"""LPIPS metric wrapper.

Grayscale handling per Architecture.md §7: replicate the single channel
to 3 channels and map [0,1] -> [-1,1] before calling LPIPS (this
approach must be *validated*, not just assumed — see
tests/test_metrics.py and docs/experiment_log.md for the check).

The `lpips` package is an optional dependency: if it is not installed,
`LPIPSMetric.available` is False and callers should skip/flag LPIPS
rather than crash, since PSNR/SSIM must still be reportable.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    import lpips as _lpips_pkg

    _HAS_LPIPS = True
except ImportError:  # pragma: no cover
    _HAS_LPIPS = False


class LPIPSMetric:
    def __init__(self, net: str = "alex", device: str | torch.device = "cpu"):
        self.available = _HAS_LPIPS
        self.device = torch.device(device)
        self._model = None
        if self.available:
            self._model = _lpips_pkg.LPIPS(net=net).to(self.device)
            self._model.eval()

    @staticmethod
    def _to_lpips_tensor(arr: np.ndarray) -> torch.Tensor:
        # (H, W) in [0,1] -> (1, 3, H, W) in [-1, 1]
        t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        t = t.repeat(1, 3, 1, 1)
        return t * 2.0 - 1.0

    @torch.no_grad()
    def __call__(self, pred: np.ndarray, target: np.ndarray) -> float | None:
        if not self.available:
            return None
        pred = np.clip(pred, 0.0, 1.0)
        target = np.clip(target, 0.0, 1.0)
        p = self._to_lpips_tensor(pred).to(self.device)
        t = self._to_lpips_tensor(target).to(self.device)
        return float(self._model(p, t).item())
