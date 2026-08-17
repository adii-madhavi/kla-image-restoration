"""Tests for supported resolutions and the 2x output-scale contract
(Architecture.md §3-4: input=[B,1,H,W], output=[B,1,2H,2W])."""
from __future__ import annotations

import pytest
import torch

from src.models import build_model


@pytest.mark.parametrize("model_name", ["bicubic", "residual_sr", "restoration_candidate"])
@pytest.mark.parametrize("hw", [(128, 128), (256, 256)])  # documented degraded resolutions
def test_output_shape_is_2x_input(model_name, hw):
    h, w = hw
    cfg = {"name": model_name, "input_channels": 1, "scale": 2}
    if model_name == "residual_sr":
        cfg.update(n_features=8, n_blocks=1)
    if model_name == "restoration_candidate":
        cfg.update(n_features=8, n_groups=1, n_blocks_per_group=1, reduction=4)

    model = build_model(cfg)
    model.eval()
    x = torch.rand(1, 1, h, w)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, h * 2, w * 2)


def test_batch_dimension_preserved():
    model = build_model({"name": "residual_sr", "input_channels": 1, "scale": 2, "n_features": 8, "n_blocks": 1})
    model.eval()
    x = torch.rand(4, 1, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (4, 1, 128, 128)
