# KLA Image Restoration — SEMICON India Hackathon 2026

**AI-Based Restoration of Degraded Images for Semiconductor Inspection**

Learns the transform from a degraded, noisy, low-resolution grayscale
image (**NoisyLR**) to its clean, full-resolution ground-truth image
(**GT**), for KLA's SEMICON India Hackathon 2026 problem statement.

```
NoisyLR input  --->  learned restoration model  --->  restored high-resolution output
```

Official benchmark degradation mechanisms (order not disclosed, may
vary, and may combine):

1. Additive Gaussian noise
2. Multiplicative speckle noise
3. Spatial downsampling

See `docs/model_card.md` for the model details and `Source_of_Truth.md`
-style project notes retained under the original planning docs for the
full requirement provenance.

---

## 1. Problem

- **Input:** grayscale degraded image, `128x128` or `256x256`. May
  contain values outside `[0,1]` — this is intentional, not a bug.
- **Output / GT:** grayscale clean image, `256x256` or `512x512`
  respectively (**2x** the input resolution). Always in `[0,1]`.
- **Goal:** faithful restoration (not generic enhancement) that is
  robust to unfamiliar image content (OOD) and to combined/unordered
  degradation.

## 2. Repository structure

```
kla-image-restoration/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── train.py                 # reproducible training entrypoint
├── inference.py              # standalone evaluator-facing inference script
├── evaluate.py                # standalone PSNR/SSIM/LPIPS evaluation
├── benchmark.py                # end-to-end runtime benchmark (H100-style timing)
├── configs/                    # one YAML per run: baseline / candidates / final
├── src/
│   ├── data/                    # io, dataset/pairing, split, augment, synthetic degradation
│   ├── models/                   # bicubic baseline, residual_sr, restoration_candidate
│   ├── losses/                    # charbonnier, ssim, gradient losses (composable)
│   ├── metrics/                    # psnr, ssim, lpips, aggregate reporting
│   ├── engine/                      # trainer, validator, checkpoint
│   └── utils/                        # seed, device, logging, config, timing
├── scripts/                    # audit_dataset, make_split, overfit_smoke_test,
│                                 generate_figures, clean_env_test.sh, package_submission
├── tests/                       # pytest unit/integration tests
├── splits/                       # serialized deterministic train/val split
├── weights/                       # trained checkpoint goes here (weights/final.pt)
├── results/                        # metrics/ figures/ samples/ produced by runs
├── presentation/                    # slide deck source + exported PDF
└── docs/                             # experiment_log, external_resources, model_card
```

## 3. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python >= 3.10, PyTorch >= 2.1. CUDA GPU recommended for training;
inference benchmarking target is an NVIDIA H100 (organizer-controlled),
but `inference.py` also runs on CPU/any CUDA GPU for development.

## 4. Dataset expectations

Point `configs/*.yaml -> data.degraded_dir / data.gt_dir` at the
official KLA-supplied paired dataset:

```
data/train/degraded/<name>.png   # or .npy / .tif — see src/data/io.py
data/train/gt/<name>.png
```

Pairing is by matching filename stem; a `degraded/<name>.png` without
a matching `gt/<name>.png` (or vice versa) raises a `PairingError`
rather than being silently dropped.

**Before training, audit the real data:**

```bash
python scripts/audit_dataset.py --degraded_dir data/train/degraded --gt_dir data/train/gt --out docs/data_audit.json
python scripts/make_split.py --degraded_dir data/train/degraded --gt_dir data/train/gt --out splits/split_seed_2026.json
```

This measures (not assumes) file format, dtype, dimensions, value
range, `%<0` / `%>1` in the degraded images, and duplicate risk —
required before deciding the `clip_input` preprocessing policy in
`configs/final.yaml`.

## 5. Training

```bash
python train.py --config configs/final.yaml
```

- Reads one resolved YAML config (model / loss / optimizer / data /
  augmentation / seed) — the same config `inference.py` reads, so
  training and inference never silently diverge.
- Writes logs, the training curve CSV, and checkpoints (`best.pt`,
  `last.pt`) to `results/<config_name>/`.
- Sanity-check any new model/loss combination first:

```bash
python scripts/overfit_smoke_test.py --config configs/residual_candidate.yaml \
    --degraded_dir data/train/degraded --gt_dir data/train/gt --n_pairs 2 --steps 300
```

Copy the chosen checkpoint to `weights/final.pt` before packaging a
submission (see `weights/README.md`).

## 6. Inference (standalone, evaluator-facing)

```bash
python inference.py --input_dir /path/to/noisy_lr_images --output_dir /path/to/restored_images
```

- No manual source edits required; weights/config default to
  `weights/final.pt` / `configs/final.yaml` and can be overridden with
  `--weights` / `--config`.
- Batches images (`--batch_size`, default from config); falls back
  gracefully to smaller batches if needed.
- Times every pipeline stage (disk read, preprocess, host-to-device,
  model execution, device-to-host, postprocess, disk write) and writes
  `<output_dir>/_runtime_report.json`.
- Output encoding: `--output_encoding uint8` (default, PNG, clipped to
  `[0,1]` — intentional, since PNG cannot represent out-of-range
  values) or `float32` (`.npy`, full precision, no clipping).

## 7. Evaluation

```bash
python evaluate.py --pred_dir results/samples/restored --gt_dir data/val/gt --out results/metrics/eval_report.json
```

Computes PSNR (`data_range=1.0`), SSIM (grayscale), and LPIPS
(AlexNet backbone; grayscale replicated to 3 channels and mapped
`[0,1] -> [-1,1]`) per image and averaged. `--no_lpips` skips LPIPS if
the optional `lpips` package is unavailable.

## 8. Runtime benchmark

```bash
python benchmark.py --input_dir data/val/degraded --config configs/final.yaml \
    --weights weights/final.pt --batch_sizes 1,4,8 --warmup 5
```

Measures the full pipeline (not just the forward pass) per batch size,
with a warm-up policy, and writes `results/metrics/runtime_benchmark.json`.

## 9. Tests

```bash
pytest -q tests/
```

Covers file I/O, pairing/validation, tensor-shape contracts across
model architectures, metric sanity (identical-image and
perturbed-image cases), model forward/gradient checks, and an
end-to-end `inference.py` CLI contract test.

## 10. Clean-environment test

Before every submission freeze, from a fresh clone:

```bash
bash scripts/clean_env_test.sh weights/final.pt configs/final.yaml
```

Installs dependencies into a throwaway venv, runs the test suite, and
runs `inference.py` end-to-end on synthetic images, checking output
dimensions and finiteness.

## 11. Packaging the submission

```bash
python scripts/package_submission.py --out submission.zip
```

Bundles README, requirements, `train.py`/`inference.py`/`evaluate.py`/
`benchmark.py`, `configs/`, `src/`, `scripts/`, `tests/`, `splits/`,
`weights/`, `results/`, `presentation/`, `docs/` — excluding caches and
local venvs.

## 12. Output contract

- Restored images are saved as single-channel PNG (`uint8`, values
  clipped to `[0,1]` then scaled to `0..255`) by default, matching the
  GT encoding. KLA does not apply clipping/normalization on its side —
  the output-range decision is made explicitly inside this repo, not
  left ambiguous.
- Filenames match the input stem (`<input_name>.png`).
- Output spatial size is exactly `2x` the input in both dimensions.

## 13. Results

Populate `results/metrics/` (PSNR/SSIM/LPIPS, ID vs OOD, runtime) and
`results/figures/` (Degraded | Restored | Ground Truth comparisons via
`scripts/generate_figures.py`) from real runs before citing numbers in
the presentation. See `docs/experiment_log.md` for the experiment
tracking template and current status (mostly TBD until the official
dataset is audited and training is run).

## 14. Hardware

Development: any CPU/CUDA machine. Official Phase 1 benchmarking: NVIDIA
H100. `src/utils/device.py` reports the exact GPU/driver context
alongside every runtime number.

## 15. External resource disclosure

See `docs/external_resources.md`. Empty until an external dataset or
pretrained model is actually adopted for the final solution — every
such resource must be disclosed with name, source, license, and
paper/model/dataset card before submission.

## 16. Limitations

- Cannot recover information genuinely destroyed by degradation; this
  is a regularized inverse-problem estimate, not exact inversion.
- Robustness outside the training/augmentation degradation coverage
  (severity, combinations) is not guaranteed.
- Restoration models can hallucinate plausible-looking structure not
  present in the ground truth.
- Hidden-test score is unknown until scored by KLA; ID/OOD validation
  numbers reported here are not a guarantee of leaderboard performance.
- The exact official KLA metric-combination weights are confidential
  and are not reproduced or guessed anywhere in this repository.

## 17. Source-of-truth / conflict notes

This repository was built against a set of internal planning documents
(retained references: problem/architecture/phases/evaluation/
submission/design notes) that reconcile several source conflicts in the
original KLA materials — most notably the presentation slide-count
(6-7 vs "8-9") and format (PPT/PPTX vs PDF) wording conflicts. The
working rule adopted throughout this repo is the stricter constraint
(6-7 slides, PDF final upload) — **always re-check the live portal
before final submission**, since organizer instructions can be updated
after this repository was built.
