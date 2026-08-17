#!/usr/bin/env python
"""Assemble the final Phase 1 submission zip.

Bundles exactly what Submission_and_Repository.md §1/§7 requires:
README, standalone evaluation/inference scripts, training script,
trained weights, restored output samples, requirements.txt, configs,
src/, docs/ (including external resource disclosure) — and nothing
that shouldn't be public (no local data, no venvs, no caches).

    python scripts/package_submission.py --out submission.zip
"""
from __future__ import annotations

import argparse
import fnmatch
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging import get_logger  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

INCLUDE_PATHS = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "pyproject.toml",
    "train.py",
    "inference.py",
    "evaluate.py",
    "benchmark.py",
    "configs",
    "src",
    "scripts",
    "tests",
    "splits",
    "weights",
    "results",
    "presentation",
    "docs",
]

EXCLUDE_PATTERNS = [
    "*.pyc",
    "__pycache__",
    "*.pyo",
    ".pytest_cache",
    "*.DS_Store",
    ".clean_env_test_venv",
    "*.ipynb_checkpoints",
]


def should_exclude(path: Path) -> bool:
    return any(fnmatch.fnmatch(part, pat) for part in path.parts for pat in EXCLUDE_PATTERNS)


def main() -> None:
    p = argparse.ArgumentParser(description="Package the repository into a Phase 1 submission zip.")
    p.add_argument("--out", default="submission.zip")
    args = p.parse_args()

    logger = get_logger("kla.package_submission")
    out_path = REPO_ROOT / args.out

    n_files = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in INCLUDE_PATHS:
            full = REPO_ROOT / rel
            if not full.exists():
                logger.warning("Skipping missing path: %s", rel)
                continue
            if full.is_file():
                if should_exclude(full):
                    continue
                zf.write(full, arcname=full.relative_to(REPO_ROOT))
                n_files += 1
                continue
            for f in full.rglob("*"):
                if f.is_file() and not should_exclude(f):
                    zf.write(f, arcname=f.relative_to(REPO_ROOT))
                    n_files += 1

    logger.info("Wrote %s (%d files).", out_path, n_files)
    logger.info(
        "Before uploading: re-check the portal naming convention "
        "(current template pattern: 'Team Name_PSNo', e.g. 'i4C_PS01') "
        "and the six/seven slide PDF requirement — see Source_of_Truth.md."
    )


if __name__ == "__main__":
    main()
