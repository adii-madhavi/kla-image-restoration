#!/usr/bin/env bash
# Clean-environment test (Architecture.md §15, Submission_and_Repository.md §11).
#
# Verifies, from a fresh virtual environment, that:
#   - dependencies install
#   - the model/checkpoint import and load
#   - the evaluator (inference.py) runs end-to-end on a tiny synthetic
#     directory of images
#   - outputs have the expected dimensions
#   - no NaN/Inf appears in the output
#   - the process exits 0
#
# Usage:
#   bash scripts/clean_env_test.sh [path/to/weights.pt] [path/to/config.yaml]
#
# Run this from the repository root, in a fresh clone, before every
# submission freeze.
set -euo pipefail

WEIGHTS="${1:-weights/final.pt}"
CONFIG="${2:-configs/final.yaml}"
VENV_DIR=".clean_env_test_venv"
TMP_INPUT="$(mktemp -d)"
TMP_OUTPUT="$(mktemp -d)"

cleanup() {
  rm -rf "${TMP_INPUT}" "${TMP_OUTPUT}"
}
trap cleanup EXIT

echo "== 1. Creating a fresh virtual environment =="
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "== 2. Installing dependencies =="
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

echo "== 3. Running unit tests =="
pytest -q tests/

echo "== 4. Generating a tiny synthetic input directory =="
python3 - "$TMP_INPUT" <<'PYEOF'
import sys
from pathlib import Path

import numpy as np
from PIL import Image

out_dir = Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
for i in range(3):
    arr = (rng.random((128, 128)) * 255).astype("uint8")
    Image.fromarray(arr, mode="L").save(out_dir / f"sample_{i:03d}.png")
print(f"Wrote 3 synthetic 128x128 images to {out_dir}")
PYEOF

echo "== 5. Running inference.py end-to-end =="
python3 inference.py --input_dir "${TMP_INPUT}" --output_dir "${TMP_OUTPUT}" --config "${CONFIG}" --weights "${WEIGHTS}"

echo "== 6. Checking output dimensions and NaN/Inf =="
python3 - "$TMP_OUTPUT" <<'PYEOF'
import sys
from pathlib import Path

import numpy as np
from PIL import Image

out_dir = Path(sys.argv[1])
files = sorted(out_dir.glob("*.png"))
assert files, f"No output images found in {out_dir}"
for f in files:
    arr = np.array(Image.open(f)).astype(np.float32)
    assert arr.shape == (256, 256), f"Unexpected output shape {arr.shape} for {f.name} (expected 256x256 for a 128x128 input)"
    assert np.isfinite(arr).all(), f"NaN/Inf detected in {f.name}"
print(f"Checked {len(files)} output image(s): shapes and finiteness OK.")
PYEOF

echo "== Clean environment test PASSED =="
