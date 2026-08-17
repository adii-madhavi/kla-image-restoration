"""Checkpoint save/load.

Preserves enough to reconstruct the exact inference configuration
(Architecture.md §10): model_state_dict, optimizer/scheduler state,
epoch, validation metrics, the resolved config, the seed, and the git
commit hash when available.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Optional

import torch


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    metrics: Optional[dict] = None,
    config: Optional[dict] = None,
    seed: Optional[int] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "metrics": metrics or {},
        "config": dict(config) if config is not None else {},
        "seed": seed,
        "git_commit": _git_commit(),
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model_weights(model: torch.nn.Module, path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """Load only the model weights (used by inference.py, which should
    not need optimizer/scheduler state)."""
    ckpt = load_checkpoint(path, map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt
