"""Final restoration candidate.

Deeper RCAN-style network (Residual Channel Attention Blocks grouped
into residual groups) plus:

- a bicubic global skip (same rationale as ResidualSR),
- an optional lightweight frequency-domain branch (FFT amplitude
  features) that can be toggled on/off from config to run the
  "spatial-only vs spatial+frequency" ablation requested in
  Architecture.md §"Frequency branch".

No degradation type/order is encoded explicitly (not required by
Architecture.md §"Architecture specialization") — the network is meant
to implicitly handle Gaussian noise, speckle noise and downsampling
(and their combinations, in any order) through channel-attention
capacity rather than degradation-specific branches, keeping it a single
model per Project.md §3's "do not implement three independent models"
requirement.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.restoration_blocks import PixelShuffleUpsampler, RCAB


class ResidualGroup(nn.Module):
    def __init__(self, channels: int, n_blocks: int, reduction: int = 16, res_scale: float = 1.0):
        super().__init__()
        self.blocks = nn.Sequential(*[RCAB(channels, reduction, res_scale) for _ in range(n_blocks)])
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(self.blocks(x))


class FrequencyBranch(nn.Module):
    """Cheap frequency-domain prior: real FFT magnitude of the input,
    log-compressed and fed through a small conv stack, then fused with
    the spatial features. Optional — ablated against spatial-only."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.rfft2(x, norm="ortho")
        mag = torch.log1p(torch.abs(fft))
        # pad/crop magnitude spectrum back to spatial H,W for a conv branch
        mag = F.interpolate(mag, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.proj(mag)


class RestorationCandidate(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        scale: int = 2,
        n_features: int = 64,
        n_groups: int = 4,
        n_blocks_per_group: int = 6,
        reduction: int = 16,
        res_scale: float = 0.2,
        use_frequency_branch: bool = False,
        clip_output: bool = True,
    ):
        super().__init__()
        self.scale = scale
        self.clip_output = clip_output
        self.use_frequency_branch = use_frequency_branch

        self.head = nn.Conv2d(input_channels, n_features, 3, padding=1)
        self.groups = nn.ModuleList(
            [ResidualGroup(n_features, n_blocks_per_group, reduction, res_scale) for _ in range(n_groups)]
        )
        self.body_tail = nn.Conv2d(n_features, n_features, 3, padding=1)

        if use_frequency_branch:
            self.freq_branch = FrequencyBranch(input_channels, n_features)
            self.fuse = nn.Conv2d(n_features * 2, n_features, 1)

        self.upsampler = PixelShuffleUpsampler(n_features, scale)
        self.tail = nn.Conv2d(n_features, input_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)

        feat = self.head(x)
        body_in = feat
        for group in self.groups:
            body_in = group(body_in)
        body_out = self.body_tail(body_in)
        feat = feat + body_out

        if self.use_frequency_branch:
            freq_feat = self.freq_branch(x)
            feat = self.fuse(torch.cat([feat, freq_feat], dim=1))

        up = self.upsampler(feat)
        out = self.tail(up) + skip

        if self.clip_output:
            out = out.clamp(0.0, 1.0)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
