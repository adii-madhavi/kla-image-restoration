"""Stage-level timing utilities.

The KLA benchmark counts the *entire* pipeline (disk read, preprocessing,
H2D transfer, model execution, D2H transfer, postprocessing, disk write),
not just the forward pass (Architecture.md §14, Project.md §6.9).
inference.py and benchmark.py both use `StageTimer` so every reported
number is measured the same way.
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

import torch


class StageTimer:
    """Accumulates wall-clock time per named pipeline stage across many
    images/batches, with correct CUDA synchronization so GPU-async work
    is actually counted."""

    STAGES = (
        "disk_read",
        "preprocess",
        "host_to_device",
        "model_execution",
        "device_to_host",
        "postprocess",
        "disk_write",
    )

    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    def _sync(self) -> None:
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def track(self, stage: str):
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed = time.perf_counter() - start
            self.totals[stage] += elapsed
            self.counts[stage] += 1

    def summary(self, n_images: int) -> dict:
        total = sum(self.totals.values())
        out = {
            "n_images": n_images,
            "total_seconds": total,
            "ms_per_image_total": (total / n_images * 1000) if n_images else None,
        }
        for stage in self.STAGES:
            secs = self.totals.get(stage, 0.0)
            out[f"{stage}_seconds"] = secs
            out[f"{stage}_ms_per_image"] = (secs / n_images * 1000) if n_images else None
        return out
