"""Model registry.

All models share the same tensor contract (Architecture.md §4):

    input  = [B, 1, H, W]
    output = [B, 1, scale*H, scale*W]

so train.py / inference.py can swap architectures purely through config.
"""
from __future__ import annotations

from src.models.bicubic import BicubicBaseline
from src.models.residual_sr import ResidualSR
from src.models.restoration_candidate import RestorationCandidate

MODEL_REGISTRY = {
    "bicubic": BicubicBaseline,
    "residual_sr": ResidualSR,
    "restoration_candidate": RestorationCandidate,
}


def build_model(cfg: dict):
    name = cfg["name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return MODEL_REGISTRY[name](**kwargs)
