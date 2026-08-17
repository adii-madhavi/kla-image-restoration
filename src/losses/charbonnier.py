"""Charbonnier loss: a differentiable, robust approximation of L1 that
avoids the zero-gradient/non-smoothness issue of |x| at x=0. Common
default pixel loss for restoration/SR networks."""
from __future__ import annotations

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps2)
        return loss.mean()
