"""Differentiable SSIM loss (1 - SSIM), grayscale, Gaussian window.
Used as the optional structural-fidelity term in the composite loss —
directly optimizes toward one of the three official reported metrics
rather than only proxying it through a pixel loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_2d = g.outer(g)
    return window_2d.unsqueeze(0).unsqueeze(0)


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.register_buffer("window", _gaussian_window(window_size, sigma), persistent=False)
        self.c1 = (0.01 * data_range) ** 2
        self.c2 = (0.03 * data_range) ** 2

    def _ssim_map(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        window = self.window.to(pred.device, pred.dtype)
        pad = self.window_size // 2

        mu_p = F.conv2d(pred, window, padding=pad)
        mu_t = F.conv2d(target, window, padding=pad)
        mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t

        sigma_p2 = F.conv2d(pred * pred, window, padding=pad) - mu_p2
        sigma_t2 = F.conv2d(target * target, window, padding=pad) - mu_t2
        sigma_pt = F.conv2d(pred * target, window, padding=pad) - mu_pt

        numerator = (2 * mu_pt + self.c1) * (2 * sigma_pt + self.c2)
        denominator = (mu_p2 + mu_t2 + self.c1) * (sigma_p2 + sigma_t2 + self.c2)
        return numerator / denominator.clamp_min(1e-12)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Force fp32 regardless of an ambient AMP autocast context: C1/C2
        # and the 1e-12 denominator floor are tuned for fp32 and silently
        # underflow to exactly 0 in fp16 (smallest fp16 subnormal is
        # ~6e-8), which turns the SSIM division into a division-by-zero
        # producing inf loss - this is not hypothetical, it reproduces on
        # every real-dataset training run under precision: amp.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            ssim_map = self._ssim_map(pred.float(), target.float())
            return 1.0 - ssim_map.mean()
