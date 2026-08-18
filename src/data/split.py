"""Deterministic train/validation splitting.

The split is computed once from the stems shared by degraded/GT
directories and serialized to splits/split_seed_<seed>.json so every
subsequent experiment reuses the exact same partition (Architecture.md
§"src/data/split.py").
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from src.data.io import list_images
from src.utils.logging import read_json, write_json


def make_split(
    degraded_dir: str | Path,
    gt_dir: str | Path,
    seed: int = 2026,
    val_fraction: float = 0.1,
    group_fn: Optional[callable] = None,
) -> dict:
    """Build a deterministic split.

    Args:
        group_fn: optional callable(stem) -> group_id. When metadata
            permits source/group identification (Project.md §4 item 10),
            splitting by group avoids near-duplicate leakage between
            train and validation. Defaults to treating every image as
            its own group.
    """
    degraded_stems = {p.stem for p in list_images(degraded_dir)}
    gt_stems = {p.stem for p in list_images(gt_dir)}
    stems = sorted(degraded_stems & gt_stems)

    if group_fn is None:
        group_fn = lambda stem: stem  # noqa: E731

    groups: dict[str, list[str]] = {}
    for stem in stems:
        groups.setdefault(group_fn(stem), []).append(stem)

    group_ids = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n_val_groups = max(1, int(round(len(group_ids) * val_fraction)))
    val_groups = set(group_ids[:n_val_groups])

    train_stems, val_stems = [], []
    for gid, members in groups.items():
        (val_stems if gid in val_groups else train_stems).extend(members)

    return {
        "seed": seed,
        "val_fraction": val_fraction,
        "n_total": len(stems),
        "n_train": len(train_stems),
        "n_val": len(val_stems),
        "train": sorted(train_stems),
        "val": sorted(val_stems),
    }


def save_split(split: dict, path: str | Path) -> None:
    write_json(path, split)


def load_split(path: str | Path) -> dict:
    return read_json(path)
