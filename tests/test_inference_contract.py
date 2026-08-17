"""CLI/output-contract tests for inference.py — the artifact KLA will
actually run (Architecture.md §12 test_inference_contract.py).

Uses the parameter-free bicubic config so the test needs no trained
weights and runs fast, while still exercising the full CLI path:
input_dir/output_dir discovery, batching, saving, and the runtime
report.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_inference_cli_end_to_end(tiny_input_dir: Path, tmp_path: Path):
    output_dir = tmp_path / "restored"
    weights_path = tmp_path / "nonexistent_weights.pt"  # bicubic doesn't need real weights

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "inference.py"),
            "--input_dir",
            str(tiny_input_dir),
            "--output_dir",
            str(output_dir),
            "--config",
            str(REPO_ROOT / "configs" / "baseline.yaml"),
            "--weights",
            str(weights_path),
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"inference.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    out_files = sorted(output_dir.glob("*.png"))
    in_files = sorted(tiny_input_dir.glob("*.png"))
    assert len(out_files) == len(in_files) == 2

    for in_f, out_f in zip(in_files, out_files):
        in_arr = np.array(Image.open(in_f))
        out_arr = np.array(Image.open(out_f))
        assert out_arr.shape == (in_arr.shape[0] * 2, in_arr.shape[1] * 2)
        assert np.isfinite(out_arr.astype(np.float32)).all()

    runtime_report = output_dir / "_runtime_report.json"
    assert runtime_report.exists()
    report = json.loads(runtime_report.read_text())
    assert report["n_images"] == 2
    assert report["total_seconds"] >= 0


def test_inference_cli_requires_no_manual_edits(tiny_input_dir: Path, tmp_path: Path):
    """The CLI must work purely from arguments — no hidden state files
    required beyond the config/weights that ship in the repo."""
    output_dir = tmp_path / "restored2"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "inference.py"),
            "--input_dir",
            str(tiny_input_dir),
            "--output_dir",
            str(output_dir),
            "--config",
            str(REPO_ROOT / "configs" / "baseline.yaml"),
            "--weights",
            str(tmp_path / "missing.pt"),
            "--device",
            "cpu",
            "--batch_size",
            "1",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0
    assert output_dir.exists()
