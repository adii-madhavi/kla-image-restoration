#!/usr/bin/env python
"""Generate and serialize the deterministic train/val split.

    python scripts/make_split.py --degraded_dir data/train/degraded --gt_dir data/train/gt --out splits/split_seed_2026.json

Run once per seed; train.py will reuse the serialized split file if it
already exists rather than regenerating it, so every experiment trains
and validates on exactly the same images (Architecture.md
§"src/data/split.py").
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.split import make_split, save_split  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Create the deterministic train/val split.")
    p.add_argument("--degraded_dir", required=True)
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--out", default="splits/split_seed_2026.json")
    args = p.parse_args()

    logger = get_logger("kla.make_split")
    split = make_split(args.degraded_dir, args.gt_dir, seed=args.seed, val_fraction=args.val_fraction)
    save_split(split, args.out)
    logger.info(
        "Wrote split to %s (%d train / %d val, seed=%d)", args.out, split["n_train"], split["n_val"], args.seed
    )


if __name__ == "__main__":
    main()
