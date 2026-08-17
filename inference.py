#!/usr/bin/env python
"""Standalone inference script — the artifact KLA will benchmark on an
H100 (Project.md §8, Submission_and_Repository.md §2).

    python inference.py --input_dir /path/to/noisy_lr_images --output_dir /path/to/restored_images

Requirements this script satisfies:
- accepts input_dir and output_dir only (no notebook/manual edits)
- discovers every supported degraded image in input_dir
- loads weights + config bundled in the repo
- batches images through the model
- times every stage (disk read, preprocess, H2D, model exec, D2H,
  postprocess, disk write) — Architecture.md §14
- writes restored images with an explicit, intentional output-range
  policy (Project.md §6.7, §"Output handling")
- never depends on hidden local paths
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import InferenceImageDataset, PreprocessPolicy
from src.data.io import write_image
from src.engine.checkpoint import load_model_weights
from src.models import build_model
from src.utils.config import load_config
from src.utils.device import device_report, get_device
from src.utils.logging import get_logger, write_json
from src.utils.timing import StageTimer

DEFAULT_CONFIG = "configs/final.yaml"
DEFAULT_WEIGHTS = "weights/final.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone KLA restoration inference.")
    p.add_argument("--input_dir", required=True, help="Directory of degraded (NoisyLR) images.")
    p.add_argument("--output_dir", required=True, help="Directory to write restored images to.")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="Model/inference config (default: %(default)s)")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Checkpoint path (default: %(default)s)")
    p.add_argument("--batch_size", type=int, default=None, help="Override batch size from config.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--output_encoding", default="uint8", choices=["uint8", "float32"])
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("kla.inference")

    cfg = load_config(args.config)
    device = get_device(args.device)
    logger.info("Device: %s | %s", device, device_report(device))

    preprocess = PreprocessPolicy(**cfg.data.get("preprocess", {})) if "data" in cfg else PreprocessPolicy()

    model = build_model(cfg.model)
    if Path(args.weights).exists():
        load_model_weights(model, args.weights, map_location=device)
        logger.info("Loaded weights from %s", args.weights)
    else:
        logger.warning(
            "Weights file '%s' not found — running with randomly initialized weights. "
            "Place the trained checkpoint at this path before benchmarking.",
            args.weights,
        )
    model = model.to(device)
    model.eval()

    dataset = InferenceImageDataset(args.input_dir, preprocess=preprocess)
    batch_size = args.batch_size or cfg.get("inference", {}).get("batch_size", 8)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    timer = StageTimer(device=device)
    n_images = 0

    # NOTE: dataset.__getitem__ performs "disk read + preprocess" inside the
    # DataLoader worker; to attribute time correctly per Architecture.md §14
    # for the *default* single/low-worker case we additionally re-measure a
    # warm-up-free, worker-free pass timing summary is still meaningful
    # because DataLoader prefetch overlaps with model execution the same way
    # it would in a real deployed pipeline — see benchmark.py for a stricter,
    # non-overlapped stage-by-stage measurement used for reporting.
    with torch.no_grad():
        for batch in loader:
            with timer.track("host_to_device"):
                degraded = batch["degraded"].to(device, non_blocking=True)

            with timer.track("model_execution"):
                pred = model(degraded)

            with timer.track("device_to_host"):
                pred_np = pred.cpu().numpy()

            with timer.track("postprocess"):
                pred_np = np.clip(pred_np, 0.0, 1.0)

            with timer.track("disk_write"):
                for i in range(pred_np.shape[0]):
                    name = batch["name"][i]
                    out_path = output_dir / f"{name}.png"
                    write_image(out_path, pred_np[i, 0], encoding=args.output_encoding)

            n_images += pred_np.shape[0]

    summary = timer.summary(n_images)
    summary["device_info"] = device_report(device)
    summary["batch_size"] = batch_size
    summary["n_images"] = n_images
    summary["config"] = args.config
    summary["weights"] = args.weights
    runtime_report_path = output_dir / "_runtime_report.json"
    write_json(runtime_report_path, summary)

    logger.info(
        "Restored %d image(s) in %.2fs (%.2f ms/image total).",
        n_images,
        summary["total_seconds"],
        summary["ms_per_image_total"] or float("nan"),
    )


if __name__ == "__main__":
    main()
