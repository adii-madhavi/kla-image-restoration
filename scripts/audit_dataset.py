#!/usr/bin/env python
"""Dataset audit script.

Produces the dataset facts Project.md §4 requires the team to know
BEFORE serious training: file format, channel count, input/target
dimensions, dtype, pairing rule, min/max input values, % below zero,
% above one, duplicate risk, and per-pair scale relationship.

    python scripts/audit_dataset.py --degraded_dir data/train/degraded --gt_dir data/train/gt --out docs/data_audit.json

Nothing here silently decides a preprocessing policy — it only measures
and reports so the decision (src/data/dataset.py PreprocessPolicy) can
be made deliberately.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.data.io import list_images, probe_file, read_image  # noqa: E402
from src.utils.logging import get_logger, write_json  # noqa: E402


def _hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def audit_directory(directory: str | Path, label: str) -> dict:
    files = list_images(directory)
    formats = Counter()
    dtypes = Counter()
    shapes = Counter()
    mins, maxs = [], []
    below_zero_fracs, above_one_fracs = [], []
    hashes = {}
    duplicate_groups = []

    for f in files:
        probe = probe_file(f)
        formats[probe.get("format", "unknown")] += 1

        arr = read_image(f)
        dtypes[str(arr.dtype)] += 1
        shapes[arr.shape] += 1
        mins.append(float(arr.min()))
        maxs.append(float(arr.max()))
        below_zero_fracs.append(float((arr < 0).mean()))
        above_one_fracs.append(float((arr > 1).mean()))

        h = _hash_array(arr)
        hashes.setdefault(h, []).append(f.name)

    for h, names in hashes.items():
        if len(names) > 1:
            duplicate_groups.append(names)

    return {
        "label": label,
        "directory": str(directory),
        "n_files": len(files),
        "file_formats": dict(formats),
        "dtypes": dict(dtypes),
        "shapes": {str(k): v for k, v in shapes.items()},
        "value_min": min(mins) if mins else None,
        "value_max": max(maxs) if maxs else None,
        "mean_fraction_below_zero": float(np.mean(below_zero_fracs)) if below_zero_fracs else None,
        "mean_fraction_above_one": float(np.mean(above_one_fracs)) if above_one_fracs else None,
        "max_fraction_below_zero": float(np.max(below_zero_fracs)) if below_zero_fracs else None,
        "max_fraction_above_one": float(np.max(above_one_fracs)) if above_one_fracs else None,
        "n_exact_duplicate_groups": len(duplicate_groups),
        "exact_duplicate_examples": duplicate_groups[:5],
    }


def audit_pairing(degraded_dir: str | Path, gt_dir: str | Path) -> dict:
    degraded_stems = {p.stem: p for p in list_images(degraded_dir)}
    gt_stems = {p.stem: p for p in list_images(gt_dir)}
    common = sorted(set(degraded_stems) & set(gt_stems))
    missing_gt = sorted(set(degraded_stems) - set(gt_stems))
    missing_deg = sorted(set(gt_stems) - set(degraded_stems))

    scale_counter = Counter()
    bad_scale_examples = []
    for stem in common[:2000]:  # cap for very large datasets
        deg = read_image(degraded_stems[stem])
        gt = read_image(gt_stems[stem])
        dh, dw = deg.shape
        gh, gw = gt.shape
        if dh == 0 or dw == 0:
            continue
        scale_h = gh / dh
        scale_w = gw / dw
        scale_counter[(round(scale_h, 2), round(scale_w, 2))] += 1
        if round(scale_h) != round(scale_w) or abs(scale_h - round(scale_h)) > 1e-6:
            bad_scale_examples.append(stem)

    return {
        "n_common_pairs": len(common),
        "n_missing_gt": len(missing_gt),
        "n_missing_degraded": len(missing_deg),
        "missing_gt_examples": missing_gt[:10],
        "missing_degraded_examples": missing_deg[:10],
        "scale_distribution": {str(k): v for k, v in scale_counter.items()},
        "non_integer_or_asymmetric_scale_examples": bad_scale_examples[:10],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Audit the official KLA dataset before training.")
    p.add_argument("--degraded_dir", required=True)
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--out", default="docs/data_audit.json")
    args = p.parse_args()

    logger = get_logger("kla.audit")
    logger.info("Auditing degraded directory: %s", args.degraded_dir)
    degraded_report = audit_directory(args.degraded_dir, "degraded")
    logger.info("Auditing GT directory: %s", args.gt_dir)
    gt_report = audit_directory(args.gt_dir, "gt")
    logger.info("Auditing pairing/scale relationship")
    pairing_report = audit_pairing(args.degraded_dir, args.gt_dir)

    report = {"degraded": degraded_report, "gt": gt_report, "pairing": pairing_report}
    write_json(args.out, report)
    logger.info("Wrote dataset audit to %s", args.out)

    print(f"\nDegraded: {degraded_report['n_files']} files, shapes={degraded_report['shapes']}")
    print(f"  value range: [{degraded_report['value_min']}, {degraded_report['value_max']}]")
    print(
        f"  mean %<0: {degraded_report['mean_fraction_below_zero']:.4%}  "
        f"mean %>1: {degraded_report['mean_fraction_above_one']:.4%}"
    )
    print(f"GT: {gt_report['n_files']} files, shapes={gt_report['shapes']}")
    print(f"  value range: [{gt_report['value_min']}, {gt_report['value_max']}]")
    print(f"Pairing: {pairing_report['n_common_pairs']} common pairs, scale={pairing_report['scale_distribution']}")


if __name__ == "__main__":
    main()
