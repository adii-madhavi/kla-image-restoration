"""Training engine.

Owns the training loop, optimizer/scheduler stepping, mixed precision,
periodic validation and checkpointing (best + last). Kept separate from
train.py (which is just config/CLI plumbing) so the loop itself is unit
-testable and reusable from scripts/overfit_smoke_test.py.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.engine.checkpoint import save_checkpoint
from src.engine.validator import validate
from src.utils.logging import CSVLogger, get_logger


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        amp: bool = True,
        grad_clip: Optional[float] = None,
        out_dir: str | Path = "results",
        config: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.amp = amp and device.type == "cuda"
        self.grad_clip = grad_clip
        self.out_dir = Path(out_dir)
        self.config = config or {}
        self.seed = seed

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.logger = get_logger("kla.trainer", self.out_dir / "train.log")
        self.csv_logger = CSVLogger(
            self.out_dir / "metrics" / "train_curve.csv",
            fieldnames=["epoch", "train_loss", "val_psnr", "val_ssim", "val_lpips", "val_loss", "lr", "epoch_seconds"],
        )
        self.best_val_psnr = float("-inf")

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        running_loss, n_batches = 0.0, 0

        for batch in loader:
            degraded = batch["degraded"].to(self.device, non_blocking=True)
            gt = batch["gt"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.amp):
                pred = self.model(degraded)
                losses = self.loss_fn(pred, gt)
                total_loss = losses["total"]

            self.scaler.scale(total_loss).backward()
            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += total_loss.item()
            n_batches += 1

        return running_loss / max(1, n_batches)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
        validate_every: int = 1,
        checkpoint_every: int = 5,
    ) -> dict:
        history = []
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            train_loss = self._train_one_epoch(train_loader)
            epoch_seconds = time.perf_counter() - t0

            val_metrics = {"val_psnr": None, "val_ssim": None, "val_lpips": None, "val_loss": None}
            if val_loader is not None and epoch % validate_every == 0:
                val_metrics = validate(self.model, val_loader, self.device, self.loss_fn)

            if self.scheduler is not None:
                self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": lr,
                "epoch_seconds": epoch_seconds,
                **val_metrics,
            }
            history.append(row)
            self.csv_logger.log(row)
            self.logger.info(
                "epoch=%d train_loss=%.5f val_psnr=%s val_ssim=%s time=%.1fs",
                epoch,
                train_loss,
                val_metrics.get("val_psnr"),
                val_metrics.get("val_ssim"),
                epoch_seconds,
            )

            weights_dir = self.out_dir / "checkpoints"
            if val_metrics.get("val_psnr") is not None and val_metrics["val_psnr"] > self.best_val_psnr:
                self.best_val_psnr = val_metrics["val_psnr"]
                save_checkpoint(
                    weights_dir / "best.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_metrics,
                    self.config,
                    self.seed,
                )
                self.logger.info("New best checkpoint (val_psnr=%.3f) saved.", self.best_val_psnr)

            if epoch % checkpoint_every == 0 or epoch == epochs:
                save_checkpoint(
                    weights_dir / "last.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_metrics,
                    self.config,
                    self.seed,
                )

        return {"history": history, "best_val_psnr": self.best_val_psnr}
