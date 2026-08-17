#!/usr/bin/env python
"""End-to-end runtime benchmark, measured the same way the evaluator
will (disk read, preprocess, H2D, model exec, D2H, postprocess, disk
write — Architecture.md §14). Unlike inference.py's DataLoader-based
path (which overlaps I/O and compute the way a real deployment would),
this script measures each stage strictly sequentially per image so the
per-stage breakdown is not confounded by prefetching, and supports a
warm-up policy plus a batch-size sweep.

    python benchmark.py --input_dir data/val/degraded --config configs/final.yaml --weights weights/final.pt --batch_sizes 1,4,8 --warmup 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import PreprocessPolicy
from src.data.io import list_images, read_image
from src.engine.checkpoint import load_model_weights
from src.models import build_model
from src.utils.config import load_config
from src.utils.device import device_report, get_device
from src.utils.logging import get_logger, write_json
from src.utils.timing import StageTimer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark end-to-end restoration throughput.")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--config", default="configs/final.yaml")
    p.add_argument("--weights", default="weights/final.pt")
    p.add_argument("--batch_sizes", default="1,4,8")
    p.add_argument("--warmup", type=int, default=5, help="Warm-up images excluded from timing.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--out", default="results/metrics/runtime_benchmark.json")
    return p.parse_args()


def run_for_batch_size(model, files, preprocess, device, batch_size, warmup):
    timer = StageTimer(device=device)
    n_timed = 0

    for start in range(0, len(files), batch_size):
        batch_files = files[start : start + batch_size]
        is_warmup = start < warmup

        arrays = []
        with (timer.track("disk_read") if not is_warmup else _null_ctx()):
            for f in batch_files:
                arr = read_image(f)
                arrays.append(arr)

        with (timer.track("preprocess") if not is_warmup else _null_ctx()):
            if preprocess.clip_input:
                arrays = [np.clip(a, preprocess.clip_min, preprocess.clip_max) for a in arrays]
            batch_np = np.stack(arrays)[:, None, :, :].astype(np.float32)
            batch_t = torch.from_numpy(batch_np)

        with (timer.track("host_to_device") if not is_warmup else _null_ctx()):
            batch_t = batch_t.to(device, non_blocking=True)

        with torch.no_grad():
            with (timer.track("model_execution") if not is_warmup else _null_ctx()):
                pred = model(batch_t)

        with (timer.track("device_to_host") if not is_warmup else _null_ctx()):
            pred_np = pred.cpu().numpy()

        with (timer.track("postprocess") if not is_warmup else _null_ctx()):
            pred_np = np.clip(pred_np, 0.0, 1.0)

        with (timer.track("disk_write") if not is_warmup else _null_ctx()):
            # Timed the same way inference.py writes, without keeping files
            # on disk for the benchmark run: encode to bytes to include
            # encoding cost without polluting the filesystem.
            from io import BytesIO

            from PIL import Image

            for i in range(pred_np.shape[0]):
                buf = BytesIO()
                out_uint8 = (pred_np[i, 0] * 255 + 0.5).astype(np.uint8)
                Image.fromarray(out_uint8, mode="L").save(buf, format="PNG")

        if not is_warmup:
            n_timed += len(batch_files)

    return timer.summary(n_timed)


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main() -> None:
    args = parse_args()
    logger = get_logger("kla.benchmark")

    cfg = load_config(args.config)
    device = get_device(args.device)
    preprocess = PreprocessPolicy(**cfg.data.get("preprocess", {})) if "data" in cfg else PreprocessPolicy()

    model = build_model(cfg.model)
    if Path(args.weights).exists():
        load_model_weights(model, args.weights, map_location=device)
    else:
        logger.warning("Weights not found at %s — benchmarking randomly initialized model.", args.weights)
    model = model.to(device).eval()

    files = list_images(args.input_dir)
    logger.info("Found %d image(s) in %s", len(files), args.input_dir)

    results = {}
    for bs in [int(x) for x in args.batch_sizes.split(",")]:
        logger.info("Benchmarking batch_size=%d", bs)
        results[f"batch_size_{bs}"] = run_for_batch_size(model, files, preprocess, device, bs, args.warmup)

    report = {
        "device_info": device_report(device),
        "warmup_images": args.warmup,
        "config": args.config,
        "weights": args.weights,
        "results_by_batch_size": results,
    }
    write_json(args.out, report)
    logger.info("Wrote runtime benchmark report to %s", args.out)


if __name__ == "__main__":
    main()
