"""Sobel-gradient edge-preservation loss.

Semiconductor inspection images depend on faithful edges/small
structures (Project.md §2: "visually pleasing smoothing is not
enough"). This term penalizes differences in gradient magnitude between
prediction and target, discouraging over-smoothed restorations.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)

    def _gradient_mag(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x.to(x.device, x.dtype), padding=1)
        gy = F.conv2d(x, self.sobel_y.to(x.device, x.dtype), padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._gradient_mag(pred), self._gradient_mag(target))
