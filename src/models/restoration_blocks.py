"""Reusable building blocks shared by the learned restoration models."""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Standard conv-relu-conv residual block (EDSR-style, no batchnorm —
    batchnorm tends to hurt restoration/SR quality by normalizing away
    the exact intensity information we're trying to preserve)."""

    def __init__(self, channels: int, res_scale: float = 1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.act(self.conv1(x)))
        return x + out * self.res_scale


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention (RCAN-style), gives
    the network a cheap way to weight feature channels by how useful
    they are for the current degradation, without a full transformer."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w


class RCAB(nn.Module):
    """Residual Channel Attention Block: ResidualBlock + ChannelAttention."""

    def __init__(self, channels: int, reduction: int = 16, res_scale: float = 1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ca = ChannelAttention(channels, reduction)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.act(self.conv1(x)))
        out = self.ca(out)
        return x + out * self.res_scale


class PixelShuffleUpsampler(nn.Module):
    """Sub-pixel convolution upsampler for integer scale factors (2, 3, 4).
    Cheaper and less prone to checkerboard artifacts than transposed conv."""

    def __init__(self, channels: int, scale: int):
        super().__init__()
        layers = []
        if scale in (2, 3):
            layers += [nn.Conv2d(channels, channels * scale * scale, 3, padding=1), nn.PixelShuffle(scale)]
        elif scale == 4:
            for _ in range(2):
                layers += [nn.Conv2d(channels, channels * 4, 3, padding=1), nn.PixelShuffle(2)]
        else:
            raise ValueError(f"Unsupported scale for PixelShuffleUpsampler: {scale}")
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)
