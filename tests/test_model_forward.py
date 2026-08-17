"""Model tensor-contract tests: forward pass, gradient flow, parameter
count reporting, and the frequency-branch ablation toggle."""
from __future__ import annotations

import torch

from src.models import build_model
from src.models.restoration_candidate import RestorationCandidate


def test_gradients_flow_through_residual_sr():
    model = build_model({"name": "residual_sr", "input_channels": 1, "scale": 2, "n_features": 8, "n_blocks": 1})
    x = torch.rand(2, 1, 32, 32, requires_grad=False)
    target = torch.rand(2, 1, 64, 64)
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_restoration_candidate_frequency_branch_toggle():
    x = torch.rand(1, 1, 32, 32)
    spatial_only = RestorationCandidate(
        n_features=8, n_groups=1, n_blocks_per_group=1, reduction=4, use_frequency_branch=False
    )
    with_freq = RestorationCandidate(
        n_features=8, n_groups=1, n_blocks_per_group=1, reduction=4, use_frequency_branch=True
    )
    spatial_only.eval()
    with_freq.eval()
    with torch.no_grad():
        y1 = spatial_only(x)
        y2 = with_freq(x)
    assert y1.shape == y2.shape == (1, 1, 64, 64)
    assert with_freq.count_parameters() > spatial_only.count_parameters()


def test_bicubic_baseline_has_no_trainable_params():
    model = build_model({"name": "bicubic", "input_channels": 1, "scale": 2})
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable == []


def test_output_is_finite():
    model = build_model({"name": "residual_sr", "input_channels": 1, "scale": 2, "n_features": 8, "n_blocks": 1})
    model.eval()
    x = torch.rand(1, 1, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert torch.isfinite(y).all()
