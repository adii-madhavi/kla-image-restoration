#!/usr/bin/env python
"""Generate the Degraded | Restored | Ground Truth comparison figures
used as evidence assets for the presentation (Design.md §2 Slide 6,
§5 "Evidence assets that must exist before final slide assembly").

    python scripts/generate_figures.py --degraded_dir data/val/degraded --restored_dir results/samples/restored --gt_dir data/val/gt --out_dir results/figures --n 6

Uses the same crop/order convention every time (Design.md §4: "Use
identical crops in comparisons. Label every image."), and also emits a
zoomed-detail + absolute-error panel per triplet.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data.io import list_images, read_image  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def make_triplet_figure(degraded, restored, gt, name: str, out_path: Path, zoom_frac: float = 0.25) -> None:
    h, w = gt.shape
    zh, zw = int(h * zoom_frac), int(w * zoom_frac)
    y0, x0 = h // 2 - zh // 2, w // 2 - zw // 2

    error = np.abs(restored - gt)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    panels = [
        (axes[0, 0], degraded, "Degraded (NoisyLR)"),
        (axes[0, 1], restored, "Restored"),
        (axes[0, 2], gt, "Ground Truth"),
    ]
    for ax, img, title in panels:
        ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    zoom_panels = [
        (axes[1, 0], degraded, "Degraded (zoom)"),
        (axes[1, 1], restored, "Restored (zoom)"),
        (axes[1, 2], gt, "GT (zoom)"),
    ]
    for ax, img, title in zoom_panels:
        img_h, img_w = img.shape
        scale_y, scale_x = img_h / h, img_w / w
        zy0, zx0 = int(y0 * scale_y), int(x0 * scale_x)
        zyh, zxw = int(zh * scale_y), int(zw * scale_x)
        crop = img[zy0 : zy0 + zyh, zx0 : zx0 + zxw]
        ax.imshow(np.clip(crop, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(f"{name}", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # separate absolute-error panel
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    im = ax2.imshow(error, cmap="inferno", vmin=0, vmax=error.max() if error.max() > 0 else 1)
    ax2.set_title(f"{name} — |Restored - GT|", fontsize=10)
    ax2.axis("off")
    fig2.colorbar(im, ax=ax2, fraction=0.046)
    fig2.tight_layout()
    fig2.savefig(out_path.with_name(out_path.stem + "_error.png"), dpi=150)
    plt.close(fig2)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate comparison figures for the presentation.")
    p.add_argument("--degraded_dir", required=True)
    p.add_argument("--restored_dir", required=True)
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--out_dir", default="results/figures")
    p.add_argument("--n", type=int, default=6, help="Number of example triplets to render.")
    args = p.parse_args()

    logger = get_logger("kla.generate_figures")
    degraded_files = {f.stem: f for f in list_images(args.degraded_dir)}
    restored_files = {f.stem: f for f in list_images(args.restored_dir)}
    gt_files = {f.stem: f for f in list_images(args.gt_dir)}
    common = sorted(set(degraded_files) & set(restored_files) & set(gt_files))[: args.n]

    if not common:
        logger.warning("No common stems found across degraded/restored/gt directories — nothing to render.")
        return

    out_dir = Path(args.out_dir)
    for stem in common:
        degraded = read_image(degraded_files[stem])
        restored = read_image(restored_files[stem])
        gt = read_image(gt_files[stem])
        make_triplet_figure(degraded, restored, gt, stem, out_dir / f"{stem}_comparison.png")
        logger.info("Rendered figure for %s", stem)

    logger.info("Wrote %d comparison figure(s) to %s", len(common), out_dir)


if __name__ == "__main__":
    main()
