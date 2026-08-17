#!/usr/bin/env python
"""Evaluator entry point (official submission interface).

    python run.py <input-dir> <output-dir>

- Reads every ``*.npy`` file in ``<input-dir>``.
- Creates ``<output-dir>`` if it does not already exist.
- Writes one restored ``.npy`` file per input file, same filename.
- Each output is a single-channel float32 array, shape ``(H, W)``,
  values clipped to ``[0, 1]``, finite (no NaN/Inf).
- Output spatial size is exactly 2x the input in both dimensions.
- Loads the model architecture and weights bundled in this repository
  only (``configs/final.yaml`` / ``weights/final.pt``) - no internet
  access, no API keys, no additional downloads, no user interaction,
  no manual configuration required.
- Runs on CPU or CUDA automatically (uses CUDA if available).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.engine.checkpoint import load_model_weights  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "final.yaml"
WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "final.pt"


def _load_array(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] in (1, 3) else arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D (or (H,W,1)) array in {path}, got shape {arr.shape}")
    return arr


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: python {Path(sys.argv[0]).name} <input-dir> <output-dir>", file=sys.stderr)
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at {WEIGHTS_PATH}. This checkpoint must be trained and "
            "committed to the repository before submission - run.py does not download weights."
        )

    cfg = load_config(str(CONFIG_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg.model)
    load_model_weights(model, WEIGHTS_PATH, map_location=device)
    model = model.to(device)
    model.eval()

    input_files = sorted(input_dir.glob("*.npy"))
    if not input_files:
        raise FileNotFoundError(f"No .npy files found in {input_dir}")

    with torch.no_grad():
        for path in input_files:
            arr = _load_array(path)
            x = torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0).unsqueeze(0).to(device)

            pred = model(x)
            pred_np = pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            pred_np = np.nan_to_num(pred_np, nan=0.0, posinf=1.0, neginf=0.0)
            pred_np = np.clip(pred_np, 0.0, 1.0).astype(np.float32)

            np.save(output_dir / path.name, pred_np)

    print(f"Restored {len(input_files)} image(s) from {input_dir} -> {output_dir}")


if __name__ == "__main__":
    main()
