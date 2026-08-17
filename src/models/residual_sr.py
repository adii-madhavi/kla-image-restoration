"""Compact residual super-resolution / restoration network.

Step 2-3 of the model progression (Architecture.md §8): a small
EDSR-style residual CNN. It jointly denoises and upsamples in one
network (Project.md §3 explicitly warns against three independent
models for the three degradation mechanisms), and adds a bicubic
"global skip" so the network only has to learn the *correction* on top
of a reasonable geometric upsample, which is easier to optimize and
harder to make catastrophically wrong.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.restoration_blocks import PixelShuffleUpsampler, ResidualBlock


class ResidualSR(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        scale: int = 2,
        n_features: int = 64,
        n_blocks: int = 8,
        res_scale: float = 0.1,
        clip_output: bool = True,
    ):
        super().__init__()
        self.scale = scale
        self.clip_output = clip_output

        self.head = nn.Conv2d(input_channels, n_features, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(n_features, res_scale) for _ in range(n_blocks)])
        self.body_tail = nn.Conv2d(n_features, n_features, 3, padding=1)
        self.upsampler = PixelShuffleUpsampler(n_features, scale)
        self.tail = nn.Conv2d(n_features, input_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)

        feat = self.head(x)
        res = self.body_tail(self.body(feat))
        feat = feat + res
        up = self.upsampler(feat)
        out = self.tail(up) + skip

        if self.clip_output:
            out = out.clamp(0.0, 1.0)
        return out
