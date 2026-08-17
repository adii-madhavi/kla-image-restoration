#!/usr/bin/env python
"""Standalone evaluation script: PSNR / SSIM / LPIPS on restored vs GT
directories (Project.md §6.8). Independent of inference.py so it can
also score any third-party output directory.

    python evaluate.py --pred_dir results/samples/restored --gt_dir data/val/gt --out results/metrics/eval_report.json
"""
from __future__ import annotations

import argparse
import json

from src.metrics.aggregate import evaluate_directories
from src.utils.logging import get_logger, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate restored images against ground truth.")
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--out", default="results/metrics/eval_report.json")
    p.add_argument("--no_lpips", action="store_true", help="Skip LPIPS (faster, no extra dependency needed).")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("kla.evaluate")

    report = evaluate_directories(args.pred_dir, args.gt_dir, use_lpips=not args.no_lpips, device=args.device)
    write_json(args.out, report)

    logger.info(
        "Evaluated %d image(s). PSNR=%.3f SSIM=%.4f LPIPS=%s (missing predictions: %d)",
        report["n_evaluated"],
        report["mean_psnr"] or float("nan"),
        report["mean_ssim"] or float("nan"),
        report["mean_lpips"],
        report["n_missing_prediction"],
    )
    print(json.dumps({k: v for k, v in report.items() if k != "per_image"}, indent=2, default=str))


if __name__ == "__main__":
    main()
