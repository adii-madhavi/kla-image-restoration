"""Raw file I/O.

Owns reading/writing image files. Deliberately contains NO model logic
and NO silent clipping — Project.md §4 requires that the degraded-input
value range be *measured*, not assumed, and that any clipping/scaling be
an explicit, validated decision made by the caller (dataset.py /
inference.py), not something buried in a loader.

Supports the file formats the official dataset is plausibly shipped in:
PNG/TIFF/BMP/JPEG (via Pillow) and NumPy .npy (for float arrays that
already extend outside [0,1], e.g. NoisyLR values below 0 or above 1).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

SUPPORTED_IMAGE_EXTS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}
SUPPORTED_ARRAY_EXTS = {".npy"}
SUPPORTED_EXTS = SUPPORTED_IMAGE_EXTS | SUPPORTED_ARRAY_EXTS


def list_images(directory: str | Path) -> list[Path]:
    """Return sorted list of supported image files in a directory
    (non-recursive by default, deterministic order for reproducibility)."""
    directory = Path(directory)
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(files)


def read_image(path: str | Path) -> np.ndarray:
    """Read a single-channel image, preserving native dtype/range.

    Returns:
        np.ndarray of shape (H, W), dtype float32. Values are NOT
        clipped or rescaled here — a uint8 PNG is divided by 255 to make
        different source formats numerically comparable, but a .npy file
        already stored as float32 in an arbitrary range (e.g. containing
        values < 0 or > 1, as NoisyLR legitimately can) is returned as-is.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in SUPPORTED_ARRAY_EXTS:
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0] if arr.shape[-1] in (1, 3) else arr[0]
        return arr

    img = Image.open(path)
    if img.mode not in ("L", "I", "I;16", "F"):
        img = img.convert("L")
    arr = np.array(img)

    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32)

    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def write_image(path: str | Path, array: np.ndarray, encoding: str = "uint8") -> None:
    """Write a single-channel image deterministically.

    Args:
        array: (H, W) float array. Caller is responsible for having
            already applied an intentional output-range policy
            (Project.md / Architecture.md §6 postprocessing step) —
            this function does the minimal, explicit conversion for the
            chosen `encoding` and nothing more.
        encoding: "uint8" (PNG, [0,1] -> 0..255, clipped at write time
            because PNG cannot represent out-of-range values) or
            "float32" (raw .npy, no clipping, full precision preserved).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if encoding == "float32":
        np.save(path.with_suffix(".npy"), array.astype(np.float32))
        return

    if encoding == "uint8":
        clipped = np.clip(array, 0.0, 1.0)
        out = (clipped * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(out, mode="L").save(path)
        return

    raise ValueError(f"Unsupported encoding: {encoding}")


def probe_file(path: str | Path) -> dict:
    """Return format/dtype/shape facts about a file without any
    normalization decisions — used by scripts/audit_dataset.py to build
    the dataset facts required by Project.md §4."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in SUPPORTED_ARRAY_EXTS:
        arr = np.load(path)
        return {
            "path": str(path),
            "format": "npy",
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        }
    img = Image.open(path)
    return {
        "path": str(path),
        "format": img.format,
        "mode": img.mode,
        "size": img.size,  # (W, H)
    }
