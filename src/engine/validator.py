"""Validation loop: runs the model over a validation DataLoader and
aggregates PSNR/SSIM (LPIPS optional — expensive per-batch, enabled via
flag) plus the training loss for monitoring."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.metrics.lpips_metric import LPIPSMetric
from src.metrics.psnr import psnr
from src.metrics.ssim import ssim


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: Optional[torch.nn.Module] = None,
    compute_lpips: bool = False,
) -> dict:
    model.eval()
    psnr_vals, ssim_vals, lpips_vals, loss_vals = [], [], [], []
    lpips_metric = LPIPSMetric(device=device) if compute_lpips else None

    for batch in loader:
        degraded = batch["degraded"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        pred = model(degraded)

        if loss_fn is not None:
            loss_vals.append(loss_fn(pred, gt)["total"].item())

        pred_np = pred.clamp(0, 1).cpu().numpy()
        gt_np = gt.cpu().numpy()
        for i in range(pred_np.shape[0]):
            p, g = pred_np[i, 0], gt_np[i, 0]
            psnr_vals.append(psnr(p, g))
            ssim_vals.append(ssim(p, g))
            if lpips_metric is not None and lpips_metric.available:
                lpips_vals.append(lpips_metric(p, g))

    model.train()
    finite_psnr = [v for v in psnr_vals if np.isfinite(v)]
    return {
        "val_psnr": float(np.mean(finite_psnr)) if finite_psnr else None,
        "val_ssim": float(np.mean(ssim_vals)) if ssim_vals else None,
        "val_lpips": float(np.mean(lpips_vals)) if lpips_vals else None,
        "val_loss": float(np.mean(loss_vals)) if loss_vals else None,
        "n_val_images": len(psnr_vals),
    }
