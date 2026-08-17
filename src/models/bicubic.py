"""Trivial baseline: bicubic upsampling, no learning.

This is step 1 of the required model progression (Architecture.md §8):
Bicubic -> tiny residual net -> compact candidate -> lightweight
candidate -> Pareto selection. It exists purely as an evidence floor —
every learned model must beat this to justify its existence.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BicubicBaseline(nn.Module):
    """Parameter-free baseline. Does not denoise; only resizes. Clips the
    (possibly out-of-[0,1]) input at the end to satisfy the GT range
    contract, but the clip is intentional/explicit, not incidental."""

    def __init__(self, input_channels: int = 1, scale: int = 2, clip_output: bool = True):
        super().__init__()
        self.scale = scale
        self.clip_output = clip_output
        # a dummy parameter so optimizer/checkpoint code paths still work
        self._unused = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        if self.clip_output:
            out = out.clamp(0.0, 1.0)
        return out
