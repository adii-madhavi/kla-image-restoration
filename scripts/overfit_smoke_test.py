#!/usr/bin/env python
"""Overfit smoke test: train on 1-2 pairs for many steps and assert the
loss collapses toward zero and PSNR rises sharply. This is a required
architecture acceptance gate (Architecture.md §"Architecture acceptance
gate" #2 and §12) before a candidate model is taken seriously —
catches broken gradients, shape bugs, or a loss that can't actually
drive the model to fit data at all.

    python scripts/overfit_smoke_test.py --config configs/residual_candidate.yaml --degraded_dir data/train/degraded --gt_dir data/train/gt --n_pairs 2 --steps 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.dataset import PairedRestorationDataset, PreprocessPolicy  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.metrics.psnr import psnr  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Overfit 1-2 pairs as a model/loss sanity check.")
    p.add_argument("--config", required=True)
    p.add_argument("--degraded_dir", required=True)
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--n_pairs", type=int, default=2)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=None, help="Override config lr (a higher lr often overfits faster).")
    args = p.parse_args()

    logger = get_logger("kla.overfit_smoke_test")
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 2026))
    device = get_device(cfg.get("device", "auto"))

    preprocess = PreprocessPolicy(**cfg.data.get("preprocess", {}))
    full_ds = PairedRestorationDataset(
        degraded_dir=args.degraded_dir, gt_dir=args.gt_dir, scale=cfg.model.get("scale", 2), preprocess=preprocess
    )
    n_pairs = min(args.n_pairs, len(full_ds))
    subset = torch.utils.data.Subset(full_ds, list(range(n_pairs)))
    loader = DataLoader(subset, batch_size=n_pairs, shuffle=False)
    batch = next(iter(loader))
    degraded = batch["degraded"].to(device)
    gt = batch["gt"].to(device)

    model = build_model(cfg.model).to(device)
    loss_fn = build_loss(cfg.loss).to(device)
    lr = args.lr or max(cfg.optimizer.get("lr", 1e-4), 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info("Overfitting %d pair(s) for %d steps (lr=%.2e)...", n_pairs, args.steps, lr)
    first_loss, first_psnr = None, None
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        pred = model(degraded)
        losses = loss_fn(pred, gt)
        losses["total"].backward()
        optimizer.step()

        if step == 1:
            first_loss = losses["total"].item()
            first_psnr = psnr(pred.detach().clamp(0, 1)[0, 0].cpu().numpy(), gt[0, 0].cpu().numpy())

        if step % max(1, args.steps // 10) == 0 or step == args.steps:
            with torch.no_grad():
                cur_psnr = psnr(pred.detach().clamp(0, 1)[0, 0].cpu().numpy(), gt[0, 0].cpu().numpy())
            logger.info("step=%d loss=%.6f psnr=%.2f", step, losses["total"].item(), cur_psnr)

    final_loss = losses["total"].item()
    final_psnr = psnr(pred.detach().clamp(0, 1)[0, 0].cpu().numpy(), gt[0, 0].cpu().numpy())

    logger.info("Loss: %.6f -> %.6f | PSNR: %.2f -> %.2f", first_loss, final_loss, first_psnr, final_psnr)
    assert final_loss < first_loss * 0.5, (
        "Overfit smoke test FAILED: loss did not drop by at least 50%. "
        "Check model/loss/optimizer wiring before trusting this architecture."
    )
    assert final_psnr > first_psnr, "Overfit smoke test FAILED: PSNR did not improve."
    logger.info("Overfit smoke test PASSED.")


if __name__ == "__main__":
    main()
