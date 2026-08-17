"""Lightweight logging helpers: a console logger plus append-only
CSV/JSONL writers for training curves and experiment records.

Deliberately dependency-free (no wandb/tensorboard requirement) so the
clean-environment test in scripts/clean_env_test.sh only needs
requirements.txt.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping


def get_logger(name: str = "kla", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class CSVLogger:
    """Append-only CSV logger. Writes header once, then one row per call."""

    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: Mapping[str, Any]) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({k: row.get(k, "") for k in self.fieldnames})


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=str)


def read_json(path: str | Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)
