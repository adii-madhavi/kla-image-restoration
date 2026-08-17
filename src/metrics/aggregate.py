"""Aggregate per-image PSNR/SSIM/LPIPS into a run-level report.

Used by evaluate.py (standalone evaluation path) and src/engine/validator.py
(training-time validation).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.data.io import list_images, read_image
from src.metrics.lpips_metric import LPIPSMetric
from src.metrics.psnr import psnr
from src.metrics.ssim import ssim


def evaluate_pair(pred: np.ndarray, target: np.ndarray, lpips_metric: Optional[LPIPSMetric] = None) -> dict:
    result = {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
    }
    if lpips_metric is not None:
        result["lpips"] = lpips_metric(pred, target)
    return result


def evaluate_directories(
    pred_dir: str | Path,
    gt_dir: str | Path,
    use_lpips: bool = True,
    device: str = "cpu",
) -> dict:
    """Evaluate every predicted image against its matching GT by stem.
    Returns per-image results plus the mean over the set."""
    pred_files = {p.stem: p for p in list_images(pred_dir)}
    gt_files = {p.stem: p for p in list_images(gt_dir)}
    common = sorted(set(pred_files) & set(gt_files))
    missing = sorted(set(gt_files) - set(pred_files))

    lpips_metric = LPIPSMetric(device=device) if use_lpips else None

    per_image = []
    for stem in common:
        pred = read_image(pred_files[stem])
        target = read_image(gt_files[stem])
        if pred.shape != target.shape:
            per_image.append({"name": stem, "error": f"shape mismatch pred={pred.shape} gt={target.shape}"})
            continue
        metrics = evaluate_pair(pred, target, lpips_metric)
        per_image.append({"name": stem, **metrics})

    valid = [r for r in per_image if "error" not in r]
    means = {}
    for key in ("psnr", "ssim", "lpips"):
        values = [r[key] for r in valid if r.get(key) is not None and np.isfinite(r[key])]
        means[f"mean_{key}"] = float(np.mean(values)) if values else None

    return {
        "n_evaluated": len(valid),
        "n_missing_prediction": len(missing),
        "missing_prediction_examples": missing[:10],
        "lpips_available": lpips_metric.available if lpips_metric is not None else False,
        **means,
        "per_image": per_image,
    }
