"""Modular loss terms combined into a weighted total loss so ablations
can toggle one component at a time (Architecture.md §9):

    L_total = w_pixel * L_pixel + w_struct * L_struct + w_grad * L_grad

Binary cross-entropy is intentionally not offered as an option — this
is a continuous image-regression task (Source_of_Truth.md §7).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.losses.charbonnier import CharbonnierLoss
from src.losses.gradient_loss import GradientLoss
from src.losses.ssim_loss import SSIMLoss

PIXEL_LOSSES = {
    "l1": nn.L1Loss,
    "l2": nn.MSELoss,
    "charbonnier": CharbonnierLoss,
}


class CompositeLoss(nn.Module):
    """Weighted sum of pixel + structural(SSIM) + gradient loss terms.
    Any weight left at 0 disables that term (and skips its computation
    when the weight is exactly 0, to keep ablation runs cheap)."""

    def __init__(
        self,
        pixel_loss: str = "charbonnier",
        w_pixel: float = 1.0,
        w_struct: float = 0.0,
        w_grad: float = 0.0,
    ):
        super().__init__()
        if pixel_loss not in PIXEL_LOSSES:
            raise ValueError(f"Unknown pixel_loss '{pixel_loss}'. Options: {sorted(PIXEL_LOSSES)}")
        self.pixel_loss = PIXEL_LOSSES[pixel_loss]()
        self.ssim_loss = SSIMLoss() if w_struct > 0 else None
        self.gradient_loss = GradientLoss() if w_grad > 0 else None
        self.w_pixel = w_pixel
        self.w_struct = w_struct
        self.w_grad = w_grad

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        losses = {"pixel": self.pixel_loss(pred, target)}
        total = self.w_pixel * losses["pixel"]

        if self.ssim_loss is not None:
            losses["struct"] = self.ssim_loss(pred, target)
            total = total + self.w_struct * losses["struct"]

        if self.gradient_loss is not None:
            losses["grad"] = self.gradient_loss(pred, target)
            total = total + self.w_grad * losses["grad"]

        losses["total"] = total
        return losses


def build_loss(cfg: dict) -> CompositeLoss:
    return CompositeLoss(**cfg)
