#!/usr/bin/env python
"""Reproducible training entrypoint.

    python train.py --config configs/final.yaml

Reads a single YAML config (model, loss, optimizer, data, augmentation,
seed — Architecture.md §11) so the checkpoint it produces is fully
reproducible from that config alone. No hard-coded local paths
(Project.md "Essential non-negotiable rules" #10) — all paths come from
the config or CLI.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.augment import build_augment
from src.data.dataset import PairedRestorationDataset, PreprocessPolicy
from src.data.split import load_split, make_split, save_split
from src.engine.trainer import Trainer
from src.losses import build_loss
from src.models import build_model
from src.utils.config import load_config, save_config
from src.utils.device import get_device
from src.utils.logging import get_logger, write_json
from src.utils.seed import seed_worker, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the KLA restoration model.")
    p.add_argument("--config", required=True, help="Path to a YAML config, e.g. configs/final.yaml")
    p.add_argument("--out_dir", default=None, help="Override output directory (default: results/<config_name>)")
    p.add_argument("--resume", default=None, help="Optional checkpoint path to resume from")
    return p.parse_args()


def _make_stem_filtered_dataset(cfg, scale, preprocess, augment, stems):
    """Build a PairedRestorationDataset restricted to a given set of
    filename stems (train or val split)."""
    allowed = set(stems)

    class _StemFiltered(PairedRestorationDataset):
        def _build_pairs(self):
            pairs = super()._build_pairs()
            return [p for p in pairs if p[0].stem in allowed]

    return _StemFiltered(
        degraded_dir=cfg.data["degraded_dir"],
        gt_dir=cfg.data["gt_dir"],
        scale=scale,
        preprocess=preprocess,
        augment=augment,
    )


def build_dataloaders(cfg, split: dict):
    preprocess = PreprocessPolicy(**cfg.data.get("preprocess", {}))
    augment = build_augment(cfg.get("augmentation"))
    scale = cfg.model.get("scale", 2)

    train_ds = _make_stem_filtered_dataset(cfg, scale, preprocess, augment, split["train"])
    val_ds = _make_stem_filtered_dataset(cfg, scale, preprocess, None, split["val"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.optimizer.get("batch_size", 8),
        shuffle=True,
        num_workers=cfg.data.get("num_workers", 4),
        pin_memory=True,
        worker_init_fn=seed_worker,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.optimizer.get("val_batch_size", cfg.optimizer.get("batch_size", 8)),
        shuffle=False,
        num_workers=cfg.data.get("num_workers", 2),
        pin_memory=True,
    )
    return train_loader, val_loader


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = cfg.get("seed", 2026)
    set_seed(seed)

    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / Path(args.config).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "resolved_config.yaml")

    logger = get_logger("kla.train", out_dir / "train.log")
    logger.info("Loaded config from %s", args.config)

    split_path = Path(cfg.data.get("split_path", "splits/split_seed_2026.json"))
    if split_path.exists():
        split = load_split(split_path)
        logger.info("Loaded existing split from %s (%d train / %d val)", split_path, split["n_train"], split["n_val"])
    else:
        split = make_split(cfg.data["degraded_dir"], cfg.data["gt_dir"], seed=seed, val_fraction=cfg.data.get("val_fraction", 0.1))
        save_split(split, split_path)
        logger.info("Created new split at %s (%d train / %d val)", split_path, split["n_train"], split["n_val"])

    device = get_device(cfg.get("device", "auto"))
    logger.info("Using device: %s", device)

    model = build_model(cfg.model)
    loss_fn = build_loss(cfg.loss)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.get("lr", 1e-4),
        weight_decay=cfg.optimizer.get("weight_decay", 0.0),
    )
    scheduler = None
    if cfg.optimizer.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.optimizer.get("epochs", 100))

    train_loader, val_loader = build_dataloaders(cfg, split)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        amp=cfg.get("precision", "amp") == "amp",
        grad_clip=cfg.optimizer.get("grad_clip"),
        out_dir=out_dir,
        config=cfg,
        seed=seed,
    )

    if args.resume:
        from src.engine.checkpoint import load_checkpoint

        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("Resumed from %s (epoch %s)", args.resume, ckpt.get("epoch"))

    result = trainer.fit(
        train_loader,
        val_loader,
        epochs=cfg.optimizer.get("epochs", 100),
        validate_every=cfg.get("validate_every", 1),
        checkpoint_every=cfg.get("checkpoint_every", 5),
    )
    write_json(out_dir / "final_result.json", result)
    logger.info("Training complete. Best val PSNR: %.3f", result["best_val_psnr"])


if __name__ == "__main__":
    main()
