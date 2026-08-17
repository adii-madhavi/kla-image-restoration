# Phases.md

# SEMICON India Hackathon 2026 --- KLA

# Detailed Gated Implementation Plan

## 0. How phases work

This is the execution controller. **Do not ask an AI coding agent to
build the entire solution in one prompt.** Complete one set, run its
tests, pass its GO/NO-GO gate, update `memory.md`, then begin the next
set.

The detailed KLA repository blueprint is incorporated below because it
contains the exact implementation ladder, prompts, artifacts, tests and
gates. The earlier gated execution playbook is also incorporated after
the repository sets to provide explicit phase questions and acceptance
criteria.

## 1. Global phase protocol

Every phase follows:

``` text
READ CURRENT STATE
      ↓
PLAN ONLY
      ↓
IMPLEMENT ONLY AGREED SCOPE
      ↓
RUN TESTS / EXPERIMENT
      ↓
DIAGNOSE
      ↓
GO / NO-GO
      ↓
UPDATE memory.md
      ↓
NEXT PHASE
```

Never skip the gate.

------------------------------------------------------------------------

# A. Full KLA Repository Build Blueprint

# SEMICON India Hackathon 2026 --- KLA Solution & GitHub Repository Build Blueprint

## Problem

**AI-Based Restoration of Degraded Images for Semiconductor Inspection**

The objective is to build a **reproducible AI image-restoration
pipeline** that takes degraded low-resolution images containing
combinations of:

-   speckle noise
-   additive Gaussian noise
-   downsampling

and reconstructs the corresponding clean image at the expected
ground-truth resolution.

The degradation order is not disclosed and should not be hard-coded.

The hidden evaluation considers:

-   **PSNR**
-   **SSIM**
-   **LPIPS**
-   **end-to-end inference throughput**
-   **training / compute hygiene and reproducibility**

This document is intentionally structured as **sets of work**. Complete
one set, run its tests, and pass its gate before asking ChatGPT/Codex to
proceed.

Do **not** ask an AI coding agent to "build the entire hackathon
solution" in one prompt.

------------------------------------------------------------------------

# 0. Core Engineering Principles

Before the sets begin, establish these rules.

## Rule 1 --- Never hard-code assumptions that can be inspected

Infer from the official paired training set:

-   file extension
-   channel count
-   LR shape
-   GT shape
-   scale factor
-   dtype
-   naming pattern

Do not hard-code a file count, dtype or `.npy` requirement unless it is confirmed by the current official dataset.

shape `128×128`, but the implementation should still be written against
the **official data contract**, not a single observed batch.

## Rule 2 --- Preserve NoisyLR values

NoisyLR values can exist outside `[0,1]`.

Do not silently clip inputs in the data loader.

## Rule 3 --- Output handling belongs to your pipeline

The evaluator scores the files exactly as saved.

Any:

-   clipping
-   normalization
-   dtype conversion
-   shape conversion

must happen intentionally inside `inference.py`.

## Rule 4 --- Baseline before sophistication

Order:

`data correctness → metric correctness → trivial baseline → tiny model → final model`

Do not start with transformers or multi-model ensembles.

## Rule 5 --- One experimental change at a time

Each run should alter one main variable:

-   architecture
-   loss
-   augmentation
-   width/depth
-   precision
-   batch size

## Rule 6 --- Every set ends in a gate

If the gate fails:

-   diagnose
-   fix
-   rerun

Do not proceed.

------------------------------------------------------------------------

# 1. Target Repository Structure

The repository should converge toward:

``` text
kla-image-restoration/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml                  # optional
├── train.py
├── inference.py
├── evaluate.py
├── benchmark.py
├── configs/
│   ├── baseline.yaml
│   ├── edsr_lite.yaml
│   ├── naf_sr_small.yaml
│   └── final.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── io.py
│   │   ├── split.py
│   │   ├── augment.py
│   │   └── synthetic_degradation.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bicubic.py
│   │   ├── residual_sr.py
│   │   ├── naf_blocks.py
│   │   └── naf_sr.py
│   ├── losses/
│   │   ├── charbonnier.py
│   │   ├── ssim_loss.py
│   │   └── gradient_loss.py
│   ├── metrics/
│   │   ├── psnr.py
│   │   ├── ssim.py
│   │   ├── lpips_metric.py
│   │   └── aggregate.py
│   ├── engine/
│   │   ├── trainer.py
│   │   ├── validator.py
│   │   └── checkpoint.py
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       ├── device.py
│       ├── config.py
│       └── timing.py
├── scripts/
│   ├── audit_dataset.py
│   ├── make_split.py
│   ├── overfit_smoke_test.py
│   ├── generate_figures.py
│   ├── clean_env_test.sh
│   └── package_submission.py
├── tests/
│   ├── test_io.py
│   ├── test_pairing.py
│   ├── test_shapes.py
│   ├── test_metrics.py
│   ├── test_model_forward.py
│   └── test_inference_contract.py
├── splits/
│   └── split_seed_2026.json
├── weights/
│   └── README.md
├── results/
│   ├── metrics/
│   ├── figures/
│   └── samples/
├── presentation/
│   └── README.md
└── docs/
    ├── experiment_log.md
    ├── external_resources.md
    └── model_card.md
```

Do not create empty complexity merely to match this tree. Build it
incrementally.

------------------------------------------------------------------------

# SET 1 --- Repository Skeleton + Environment

## Goal

Create a minimal repository that can be cloned, installed and tested.

## Ask Codex / ChatGPT

Use a prompt similar to:

> We are building the SEMICON India Hackathon KLA image-restoration
> solution. For this step only, create the minimal repository skeleton,
> `requirements.txt`, `.gitignore`, `README.md` placeholder, `src/`
> package, `tests/`, and a small environment-check script. Do not
> implement the model or training yet. Use Python 3.10/3.11-compatible
> code and PyTorch with CUDA support. Keep dependencies minimal. Add a
> command that prints Python, PyTorch, CUDA, GPU and NumPy versions.

## Dependencies to consider initially

Keep initial dependencies small:

``` text
numpy
torch
torchvision
scikit-image
pillow
pyyaml
tqdm
pytest
```

Add LPIPS only in the metric set when needed.

Optional later:

``` text
tensorboard
lpips
opencv-python-headless
pandas
matplotlib
```

## Required outputs

-   repo skeleton
-   working virtual environment
-   version-check script
-   `pytest` starts successfully
-   README contains problem name and status

## Test

``` bash
python -c "import torch, numpy; print(torch.__version__, torch.cuda.is_available())"
pytest -q
```

## GO / NO-GO gate

Proceed only if:

-   imports succeed
-   CUDA is visible on the development GPU if a GPU exists
-   `pytest` exits successfully
-   no project module requires a hard-coded dataset path

------------------------------------------------------------------------

# SET 2 --- Dataset Audit Only

## Goal

Understand the official data before writing a model.

## What to inspect

For both training GT and NoisyLR:

-   directory names
-   number of files
-   filenames
-   file extensions
-   array shapes
-   number of channels
-   dtypes
-   min/max
-   mean/std
-   NaN count
-   Inf count
-   unique shape distribution
-   pairing consistency
-   LR→GT scale factor
-   whether every LR has exactly one GT
-   whether arrays are contiguous
-   whether there are duplicate files / near duplicates

For test data:

-   same audit except GT-dependent items

## Ask Codex / ChatGPT

> Implement only `scripts/audit_dataset.py`. It must recursively inspect
> the supplied GT and NoisyLR directories, load the official image files safely,
> match pairs by stem, report counts/shapes/dtypes/value ranges/NaN/Inf
> statistics, infer all observed LR→GT scale factors, and emit both a
> human-readable console report and `results/dataset_audit.json`. Do not
> clip or modify arrays. Add unit tests for pairing and scale-factor
> inference. Do not write model code.

## Important observed fact from supplied test package

An audit of the attached `Test_NoisyLR.zip` shows:

-   all observed arrays: `(128, 128)`
-   dtype: `float32`
-   values can fall below 0 and rise well above 1

This is consistent with the KLA note that NoisyLR can extend outside
`[0,1]`.

Do not make `128×128` a permanent model restriction.

## Suggested output schema

``` json
{
  "train": {
    "n_pairs": 0,
    "lr_shapes": {},
    "gt_shapes": {},
    "lr_dtype": {},
    "gt_dtype": {},
    "lr_global_min": null,
    "lr_global_max": null,
    "gt_global_min": null,
    "gt_global_max": null,
    "scale_factors": {}
  },
  "test": {
    "n_inputs": 0,
    "shapes": {},
    "dtype": {},
    "global_min": null,
    "global_max": null
  }
}
```

## GO / NO-GO gate

Proceed only if:

-   100% of expected train pairs match
-   no unexplained shape mismatch exists
-   scale factor is known for all training pairs
-   data type/range behavior is documented
-   NaN/Inf behavior is known
-   no loader-side clipping has been added

If any pair mismatch exists, stop and fix dataset interpretation.

------------------------------------------------------------------------

# SET 3 --- Robust I/O Layer + Pair Dataset

## Goal

Create the input/output contract before the network.

## Files

Build:

-   `src/data/io.py`
-   `src/data/dataset.py`
-   `tests/test_io.py`
-   `tests/test_pairing.py`
-   `tests/test_shapes.py`

## Data loader requirements

### Input

-   load official array format
-   cast to model tensor safely
-   convert `[H,W]` to `[1,H,W]`
-   preserve raw NoisyLR numerical range
-   preserve GT target values
-   return filename/stem

### Paired dataset

Return:

``` python
{
    "lr": lr_tensor,
    "gt": gt_tensor,
    "name": filename,
    "scale": scale
}
```

### Output saver

Must:

-   restore expected spatial shape
-   remove batch/channel dimension correctly
-   save with exact required filename
-   save the required data type/format
-   explicitly apply final range policy

## Range policies to support

Implement as config options, not hard-coded experiments:

-   `none`
-   `clamp_0_1`

The final policy is chosen by validation.

## Ask Codex / ChatGPT

> Implement the data I/O layer and paired Dataset only. Preserve NoisyLR
> values outside \[0,1\]. Support the official grayscale array/file representation and
> expose channel-first tensors to the model. Implement a save function
> that writes predictions back to the official array shape and filename.
> Add tests using temporary synthetic arrays to prove round-trip
> filename, dtype and shape behavior. Do not build a network yet.

## GO / NO-GO gate

Proceed only if:

-   unit tests pass
-   one real training pair loads correctly
-   LR tensor shape is correct
-   GT tensor shape is correct
-   save→reload preserves expected output shape
-   filename is preserved exactly
-   raw LR values outside `[0,1]` survive the loader

------------------------------------------------------------------------

# SET 4 --- Validation Split + Metric Correctness

## Goal

Make the evaluation system trustworthy before training.

## Part A --- Split

Build:

-   `scripts/make_split.py`
-   `src/data/split.py`
-   `splits/split_seed_2026.json`

Recommended starting split:

-   80--90% training
-   10--20% validation

Exact ratio is less important than:

-   fixed seed
-   zero leakage
-   frozen file list
-   good degradation diversity

### Better-than-random option

Cluster or bin images using cheap LR statistics:

-   mean
-   std
-   min/max
-   high-frequency energy
-   rough noise estimate

Then distribute bins across train/val.

This can make the validation set less accidentally easy.

## Part B --- Metrics

Implement:

-   PSNR ↑
-   SSIM ↑
-   LPIPS ↓

Optional diagnostics:

-   MAE
-   RMSE
-   gradient error

## Metric correctness concerns

### PSNR

Use the correct data range:

`data_range = 1.0`

when comparing GT and final clipped/restored values in `[0,1]`.

### SSIM

Handle grayscale correctly.

### LPIPS

LPIPS networks generally expect 3-channel images and a specific
normalized range.

For grayscale:

1.  replicate channel 3 times
2.  transform `[0,1] → [-1,1]`
3.  compute LPIPS

Do not accidentally calculate LPIPS on raw unbounded NoisyLR values.

## Metric unit tests

Must include:

1.  identical image:
    -   SSIM ≈ 1
    -   LPIPS ≈ 0
    -   PSNR very high/infinite depending implementation
2.  perturbed image:
    -   lower PSNR
    -   lower SSIM
    -   higher LPIPS

## Ask Codex / ChatGPT

> Create the fixed train/validation split utilities and metric module.
> Implement PSNR, SSIM and LPIPS with explicit grayscale handling and
> documented data ranges. Add unit tests using identical and perturbed
> synthetic images. Do not train a model yet. The test suite should
> detect an incorrect LPIPS input normalization or channel count.

## GO / NO-GO gate

Proceed only if:

-   split JSON is deterministic
-   no file is in both train and validation
-   metrics pass sanity tests
-   all three metrics run on a real validation pair
-   metric preprocessing is documented

------------------------------------------------------------------------

# SET 5 --- Baseline 0: Bicubic Restoration

## Goal

Establish the "zero-learning" score and verify the full evaluation path.

## Implementation

For each LR:

1.  resize to GT dimensions using bicubic interpolation
2.  evaluate:
    -   unclipped output if meaningful
    -   clipped `[0,1]` output
3.  compare both policies

## Files

-   `src/models/bicubic.py`
-   `evaluate.py`
-   first version of `benchmark.py`

## Required results

Write:

``` text
results/metrics/bicubic_validation.csv
results/metrics/bicubic_summary.json
```

Record:

-   PSNR
-   SSIM
-   LPIPS
-   total runtime
-   ms/image

## Ask Codex / ChatGPT

> Implement a bicubic baseline that uses the dataset-inferred
> scale/output shape. Run it through the exact same save/evaluate path
> planned for the final model. Compare no output clipping versus
> clamp-to-\[0,1\]. Save per-image and aggregate metrics. Do not add a
> neural network in this step.

## GO / NO-GO gate

Proceed only if:

-   bicubic output matches GT shape for 100% of validation samples
-   all metrics run
-   one aggregate result file exists
-   output range policy comparison is complete
-   image panels look geometrically aligned

If bicubic and GT appear shifted or misaligned, stop. The pairing/shape
pipeline is wrong.

------------------------------------------------------------------------

# SET 6 --- Tiny Neural Network + One-Pair Overfit Test

## Goal

Prove the training path can learn before starting expensive experiments.

## Model

Create a tiny residual SR network:

``` text
LR
→ conv
→ 3–5 residual blocks
→ upsample
→ conv
→ output
```

Add a bicubic/global residual path if useful.

## Overfit protocol

Use:

-   1--2 training pairs
-   no random augmentation
-   small patch or full image
-   L1/Charbonnier loss
-   several hundred iterations

Expected behavior:

-   training loss drops sharply
-   PSNR rises significantly
-   prediction becomes close to GT

## Files

-   `src/models/residual_sr.py`
-   `src/losses/charbonnier.py`
-   minimal trainer
-   `scripts/overfit_smoke_test.py`

## Ask Codex / ChatGPT

> Implement the smallest possible residual super-resolution network and
> a one-pair overfit smoke test. Do not optimize for leaderboard
> performance. The purpose is to prove data alignment, gradient flow,
> loss correctness, checkpoint save/load and reconstruction shape. Print
> loss and validation metrics during the smoke test and save one
> before/after prediction.

## GO / NO-GO gate

Proceed only if:

-   loss decreases strongly
-   model can substantially overfit 1--2 samples
-   saved checkpoint reloads and reproduces the same output
-   no shape mismatch exists
-   no NaNs occur

If the model cannot overfit one pair, do **not** build a larger model.

------------------------------------------------------------------------

# SET 7 --- Baseline 1: Compact Residual Super-Resolution Model

## Goal

Create the first serious learned baseline.

## Suggested model

A compact residual super-resolution style network:

-   32--64 feature channels
-   8--12 residual blocks
-   no batch normalization
-   PixelShuffle upsampling
-   bicubic/global residual path

Why:

-   simple
-   fast
-   strong enough to validate training strategy
-   easy to debug

## Training v1

Start with:

-   official paired data only
-   Charbonnier or L1
-   geometric augmentations only
-   AdamW or Adam
-   cosine or step scheduler
-   AMP if stable
-   checkpoint best validation model

## Required logs

Each run should write:

``` text
runs/<run_id>/
├── config.yaml
├── metrics.csv
├── best.pt
├── last.pt
├── environment.txt
└── command.txt
```

## Ask Codex / ChatGPT

> Train a compact residual baseline using only the official paired training
> data and paired geometric augmentations. Use the frozen validation
> split. Add config-driven hyperparameters, AMP, checkpointing, early
> stopping or best-model selection, CSV logging and seed control. Do not
> add synthetic degradations or advanced losses yet. Produce
> PSNR/SSIM/LPIPS and true end-to-end validation inference time.

## GO / NO-GO gate

Proceed only if:

-   learned model beats bicubic on the primary quality metrics overall
-   training is stable
-   best checkpoint reloads
-   validation metrics are reproducible within a small tolerance
-   output images do not contain obvious tiling/checkerboard artifacts

------------------------------------------------------------------------

# SET 8 --- Synthetic Degradation Calibration

## Goal

Improve hidden-test generalization without guessing unrealistic
corruption ranges.

The KLA challenge permits synthetic degraded pairs created from supplied
GT data using the official mechanisms.

## Stage 8A --- Analyze paired residuals

Estimate, as far as practical:

-   effective downsampling scale
-   local variance behavior
-   Gaussian-like residual scale
-   multiplicative/speckle relationship
-   image-wise difficulty distribution

Do not expect perfect inverse identification because degradation order
is unknown.

## Stage 8B --- Build degradation generator

Implement only:

1.  downsampling
2.  Gaussian noise
3.  speckle noise

Randomize order.

Parameter ranges should come from:

-   training pair observations
-   controlled validation sweep

not arbitrary extreme values.

## Stage 8C --- Validate generator visually and statistically

Compare synthetic NoisyLR against real NoisyLR using:

-   min/max
-   mean/std
-   histogram
-   noise residual statistics
-   frequency energy
-   side-by-side images

## Ask Codex / ChatGPT --- first prompt

> Analyze the paired training data to estimate useful degradation
> statistics without assuming the hidden order. Produce plots and JSON
> summaries only. Do not synthesize data yet.

## Ask Codex / ChatGPT --- second prompt after inspection

> Based on the measured training statistics, implement a configurable
> synthetic degradation generator using only Gaussian noise, speckle
> noise and downsampling. Randomize their order. Add deterministic tests
> and a script that compares synthetic versus official NoisyLR
> statistics. Do not retrain the final model yet.

## GO / NO-GO gate

Proceed only if:

-   generator uses only allowed mechanisms
-   output LR shapes are valid
-   no NaN/Inf
-   synthetic ranges resemble official training observations
-   paired geometry remains correct
-   generated examples look plausible

------------------------------------------------------------------------

# SET 9 --- Final Candidate Architecture: Lightweight Restoration Candidate

## Goal

Build a more expressive but throughput-aware model.

## Recommended architecture

### High-level

``` text
LR
│
├── Bicubic upsample ───────────────────────────┐
│                                               │
└── Shallow Conv
    → lightweight restoration blocks
    → optional 2-level feature hierarchy
    → upsampling head
    → reconstruction conv
                                                │
                       residual correction ─────┘
                              ↓
                         restored HR
```

## Design requirements

-   single-channel input/output
-   fully convolutional
-   no fixed `128×128` assumption
-   support inferred scale factor
-   no global full-resolution self-attention
-   residual path
-   AMP-safe
-   batchable

## Model sizes to compare

Do not tune 20 variants.

Try a small matrix:

### Small

-   width \~32
-   modest block count

### Medium

-   width \~48
-   more blocks

### Large only if justified

-   width \~64

The exact block counts should be config-driven.

## Ask Codex / ChatGPT

> Implement a compact compact restoration model with a published or justified block design for
> single-channel restoration. It must be fully convolutional, use a
> residual/bicubic skip, support the dataset-inferred upscaling factor,
> avoid fixed spatial sizes, and expose width/block count via YAML. Add
> forward-shape tests and parameter-count reporting. Do not train all
> variants yet.

## GO / NO-GO gate

Proceed only if:

-   all model variants pass forward tests
-   parameter count is recorded
-   inference works for more than one spatial shape
-   AMP forward pass works
-   no hard-coded test dimensions

------------------------------------------------------------------------

# SET 10 --- Loss Ablation

## Goal

Find a fidelity-oriented objective using controlled experiments.

## Run A

`Charbonnier`

## Run B

`Charbonnier + SSIM`

## Run C

`Charbonnier + SSIM + Gradient`

## Optional Run D

A very small perceptual term only if visual/LPIPS gains do not damage
structural fidelity.

## Gradient loss concept

Compare first derivatives:

-   horizontal gradient
-   vertical gradient

This can encourage edge preservation.

## Evaluation

For every loss experiment:

-   PSNR
-   SSIM
-   LPIPS
-   runtime
-   sample crops
-   failure count

## Ask Codex / ChatGPT

> Add SSIM loss and gradient loss as optional weighted terms. Run a
> controlled loss ablation on one fixed model/config, changing only the
> loss. Keep optimizer, seed, split, epochs and augmentation fixed.
> Produce one comparison CSV and sample panels. Do not change
> architecture during this set.

## GO / NO-GO gate

Choose a loss only if:

-   it improves the overall metric tradeoff
-   it does not create halos/ringing
-   small structures remain plausible
-   gains repeat on more than a few samples

------------------------------------------------------------------------

# SET 11 --- Synthetic Augmentation Ablation

## Goal

Determine whether synthetic pairs improve generalization.

## Experiments

### A

Official pairs only

### B

Official + conservative synthetic degradation

### C

Official + wider synthetic degradation

Do not introduce unsupported corruption types.

## Sampling strategy

Possible approach:

-   60--80% official pairs
-   20--40% synthetic pairs

Treat ratio as an experiment, not a fact.

## Ask Codex / ChatGPT

> Using the frozen final candidate model and loss, compare official-only
> training against official-plus-synthetic training. Keep total
> optimization steps similar. Test one conservative and one wider
> synthetic range derived from the audit. Produce metrics by
> degradation-difficulty bins if possible.

## GO / NO-GO gate

Keep synthetic augmentation only if it:

-   improves overall validation or hard-bin performance
-   does not materially reduce easy/in-distribution quality
-   does not create visual artifacts
-   is fully reproducible

------------------------------------------------------------------------

# SET 12 --- Model Size / Throughput Pareto Search

## Goal

Optimize for both restoration quality and H100 execution.

## Compare only a few candidates

For example:

  Model      Width   Blocks   Params Quality   Runtime
  -------- ------- -------- -------- --------- ---------
  Small         32   config   actual actual    actual
  Medium        48   config   actual actual    actual
  Large         64   config   actual actual    actual

## Runtime protocol

Benchmark:

-   batch size 1
-   realistic batch size
-   FP32
-   AMP/FP16 if numerically safe

### End-to-end timing must include

1.  directory/file discovery
2.  disk loading
3.  preprocessing
4.  CPU→GPU
5.  model
6.  GPU→CPU
7.  post-processing
8.  saving

Also separately record model-only time for diagnosis, but do not confuse
it with the official end-to-end figure.

## Ask Codex / ChatGPT

> Implement `benchmark.py` to measure both model-only and true
> end-to-end inference time. Include warmup, CUDA synchronization, image
> I/O and saving in the end-to-end measurement. Benchmark a small fixed
> set of model sizes and batch sizes. Produce CSV/JSON output and a
> quality-vs-runtime Pareto plot.

## GO / NO-GO gate

Select the final model only after reviewing:

-   PSNR
-   SSIM
-   LPIPS
-   end-to-end runtime
-   memory
-   visual quality

Do not choose the largest model automatically.

------------------------------------------------------------------------

# SET 13 --- Output Range Policy Test

## Goal

Determine what is actually saved.

Because GT is `[0,1]`, compare:

### Policy A

No clamp

### Policy B

Clamp to `[0,1]`

Potential additional policy only if justified:

### Policy C

soft bounded output / learned bounded activation

Do not use a policy merely because it "looks safer."

## Test

Run the same checkpoint under each output policy.

Record:

-   fraction of predicted pixels \< 0
-   fraction \> 1
-   PSNR
-   SSIM
-   LPIPS

## GO / NO-GO gate

Freeze the output policy in `configs/final.yaml` and document it in
README.

------------------------------------------------------------------------

# SET 14 --- Error Analysis

## Goal

Understand when the model fails.

## Compute per-image table

``` text
filename
psnr
ssim
lpips
lr_mean
lr_std
lr_min
lr_max
high_frequency_energy
```

## Select

-   best 5
-   median 5
-   worst 10

## Inspect for patterns

-   high noise
-   extreme LR range
-   repetitive texture
-   sharp thin lines
-   low contrast
-   strong downsampling loss

## Generate panels

Each:

`LR | bicubic | model | GT | absolute error`

Add zoomed crop.

## Ask Codex / ChatGPT

> Build an error-analysis script that ranks validation images by
> PSNR/SSIM/LPIPS, joins those scores with LR statistics, and exports
> representative best/median/worst visual panels plus a CSV. Do not
> alter the model in this step. Summarize recurring failure patterns
> from measurements only.

## GO / NO-GO gate

Proceed only after at least one honest failure case has been selected
for the presentation.

------------------------------------------------------------------------

# SET 15 --- Final Inference Script

## Goal

Meet the mandatory evaluator contract.

## Required CLI

At minimum:

``` bash
python inference.py \
  --input_dir /path/to/NoisyLR \
  --output_dir /path/to/restored
```

Optional arguments may include:

``` text
--weights
--config
--batch_size
--device
--amp
```

But evaluators should not need to edit source code.

## Mandatory behavior

-   discover all input files
-   stable sort
-   load every image
-   restore every image
-   save every output
-   preserve required file naming
-   create output dir if needed
-   GPU support
-   batch when shapes permit
-   helpful errors
-   progress indicator
-   deterministic inference
-   no dependency on notebook state
-   no absolute local path

## Edge cases

Test:

-   empty directory
-   non-data file in input directory
-   one image
-   400 images
-   output directory already exists
-   CPU fallback if desired
-   insufficient GPU memory → smaller batch or clear error

## Ask Codex / ChatGPT

> Harden `inference.py` as a standalone directory-to-directory evaluator
> script. It must require no code edits, load the final YAML and
> checkpoint, preserve exact basenames, process every official input,
> support CUDA and batch inference, apply the frozen output-range
> policy, and save arrays in the required official format. Add
> integration tests using temporary input/output directories.

## GO / NO-GO gate

Proceed only if:

``` bash
python inference.py --input_dir <test> --output_dir <new_empty_dir>
```

runs end-to-end with no manual edits and produces exactly one valid
output per valid input.

------------------------------------------------------------------------

# SET 16 --- Reproducibility / Clean Environment Test

## Goal

Prove another machine can run the repository.

## Create fresh environment

Example:

``` bash
python -m venv .venv_clean
source .venv_clean/bin/activate
pip install -r requirements.txt
```

Then run:

``` bash
pytest -q
python inference.py --input_dir sample_inputs --output_dir /tmp/kla_outputs
```

If training must be demonstrated:

``` bash
python train.py --config configs/baseline.yaml --max_epochs 1
```

## Capture

``` text
results/reproducibility/
├── environment.txt
├── pip_freeze.txt
├── pytest.txt
└── inference_log.txt
```

## Ask Codex / ChatGPT

> Review the repository as if you are a KLA evaluator on a clean
> machine. Identify every hidden dependency, hard-coded path, missing
> weight/config, undocumented command, or non-deterministic assumption.
> Produce a fix list first. Make changes only after I approve the list.

This "review first, edit second" pattern is important.

## GO / NO-GO gate

Proceed only if a clean environment can execute the final inference
command without editing source code.

------------------------------------------------------------------------

# SET 17 --- Test-Set Dry Run

## Goal

Run the exact final inference package on the supplied hidden-target
inputs.

Do not train on test inputs.

## Dry run checklist

-   count input files
-   count output files
-   all names match
-   all outputs load
-   all outputs have expected target resolution
-   dtype correct
-   no NaN
-   no Inf
-   final range policy applied
-   runtime captured
-   no temporary/debug files mixed into output directory

## Suggested validation script

Create:

``` bash
python scripts/validate_submission_outputs.py \
  --input_dir <test_input> \
  --output_dir <restored_output>
```

It should fail loudly if:

-   a name is missing
-   an extra name exists
-   an array cannot load
-   shape is wrong
-   dtype is wrong
-   NaN/Inf exists

## GO / NO-GO gate

Output validation must be 100% clean.

------------------------------------------------------------------------

# SET 18 --- Final Result Artifacts

## Goal

Create everything needed for the PPT/PDF from code.

## Generate

``` text
results/metrics/final_validation.csv
results/metrics/experiment_table.csv
results/metrics/runtime_summary.json
results/figures/metric_comparison.png
results/figures/runtime_quality_pareto.png
results/figures/visual_case_01.png
results/figures/visual_case_02.png
results/figures/visual_failure_case.png
results/figures/error_maps.png
results/samples/
```

## Ask Codex / ChatGPT

> Generate publication-quality result artifacts directly from the final
> experiment logs and checkpoint. Do not manually type metric values
> into plotting code. All charts/tables must read the saved CSV/JSON
> results so the presentation cannot drift from the repository evidence.

## GO / NO-GO gate

Every number used in the PPT must trace back to a machine-generated
result file.

------------------------------------------------------------------------

# SET 19 --- README + Model Card + External Resource Disclosure

## README required sections

1.  Challenge
2.  Solution summary
3.  Repository structure
4.  Environment
5.  Dataset placement
6.  Training
7.  Validation
8.  Inference
9.  Metrics
10. Runtime benchmarking
11. Final model
12. Output format
13. Reproducibility
14. External resources/licences
15. Known limitations

## Model card

Include:

-   architecture
-   input/output
-   parameter count
-   scale factor
-   loss
-   training data
-   synthetic data policy
-   seed
-   checkpoint selection
-   metrics
-   runtime
-   limitations

## External resources

For each:

``` text
Name:
Purpose:
URL:
Licence:
Pretrained weights used?:
External training data used?:
```

## Ask Codex / ChatGPT

> Draft the final README and model card strictly from the current
> repository/config/result files. Do not invent metrics, hardware or
> commands. If information is missing, insert a clearly marked TODO and
> list what evidence is needed.

## GO / NO-GO gate

A teammate who has never seen the project should be able to run
inference using only the README.

------------------------------------------------------------------------

# SET 20 --- Final Repository Audit

## Goal

Do not submit a development repository full of clutter or secrets.

## Remove / exclude

-   `.venv`
-   raw private/local data
-   giant temporary checkpoints
-   caches
-   notebook checkpoints
-   local absolute paths
-   API keys
-   tokens
-   debug dumps
-   test outputs not required
-   OS metadata such as `.DS_Store`

## Keep

-   final checkpoint or valid download instruction
-   final config
-   scripts
-   tests
-   README
-   dependency spec
-   results summary
-   presentation
-   licence disclosures

## Ask Codex / ChatGPT

> Perform a final repository audit. Do not change code yet. Produce a
> checklist of: missing submission files, secrets/large files, stale
> configs, unused code, hard-coded paths, commands that may fail,
> missing licences, and mismatch between README and actual CLI. Then we
> will fix each item one by one.

## GO / NO-GO gate

Run:

``` bash
git status
pytest -q
python inference.py --help
```

Then clone the repository into a different directory and repeat the
clean run.

------------------------------------------------------------------------

# SET 21 --- GitHub Release / Submission Package

## Goal

Freeze the exact submitted version.

## Tag

Example:

``` bash
git tag -a phase1-submission -m "SEMICON India Hackathon 2026 Phase 1"
git push origin phase1-submission
```

## Release manifest

Create:

``` text
SUBMISSION_MANIFEST.md
```

with:

-   team name
-   problem statement
-   Git commit
-   Git tag
-   final config
-   final checkpoint checksum
-   inference command
-   validation scores
-   runtime environment
-   PDF filename
-   demo URL

## Checkpoint checksum

Example:

``` bash
sha256sum weights/final.pt
```

## GO / NO-GO gate

The commit linked in the submission must be the exact commit that passed
the clean environment test.

------------------------------------------------------------------------

# 2. Recommended Experiment Ladder

Do not run arbitrary experiments. Use this ladder.

  Exp ID   Model           Data           Loss                  Purpose
  -------- --------------- -------------- --------------------- ------------------------
  B0       Bicubic         ---            ---                   non-learning floor
  B1       compact residual baseline       official       Charbonnier           learned baseline
  M1       lightweight candidate    official       Charbonnier           architecture test
  M2       lightweight candidate    official       Charb + SSIM          loss test
  M3       lightweight candidate    official       Charb + SSIM + grad   edge test
  M4       lightweight candidate    + synthetic   best loss             generalization
  M5       medium candidate     best data      best loss             capacity test
  FINAL    chosen          chosen         chosen                quality/runtime Pareto

Stop early if an experiment is clearly worse.

------------------------------------------------------------------------

# 3. Suggested Configuration File

Example shape only; replace values with measured/tuned values.

``` yaml
experiment:
  name: candidate_restoration_small
  seed: 2026

data:
  train_lr_dir: data/train/NoisyLR
  train_gt_dir: data/train/GT
  split_file: splits/split_seed_2026.json
  num_workers: 4
  patch_size_lr: 64
  preserve_lr_range: true

model:
  name: candidate_restoration
  in_channels: 1
  out_channels: 1
  width: 32
  blocks: [4, 4, 4]
  scale: auto

loss:
  charbonnier: 1.0
  ssim: 0.0
  gradient: 0.0

training:
  epochs: 100
  batch_size: 16
  optimizer: adamw
  lr: 0.0002
  weight_decay: 0.0
  amp: true

validation:
  metrics: [psnr, ssim, lpips]
  output_policy: clamp_0_1

checkpoint:
  monitor: multi_metric_policy
  save_best: true
```

Do not blindly use the numerical values above. They are scaffolding, not
KLA-provided hyperparameters.

------------------------------------------------------------------------

# 4. Checkpoint Selection Strategy

KLA has not disclosed the exact weighted combination of quality metrics.

Therefore avoid inventing a fake "KLA score."

## Safer selection approach

For each checkpoint:

1.  compute mean PSNR
2.  compute mean SSIM
3.  compute mean LPIPS
4.  inspect worst-case images
5.  record runtime

Then use a transparent decision policy.

Example:

-   discard any candidate with major regression in one primary metric
-   among remaining candidates, prefer Pareto-efficient quality/runtime
    points
-   use a simple normalized internal score only for your own ranking if
    you document that it is **not** the official KLA formula

------------------------------------------------------------------------

# 5. Inference Optimization Order

Do not optimize prematurely.

After correctness is frozen:

1.  `torch.inference_mode()`
2.  batch inference
3.  pinned memory
4.  `non_blocking=True`
5.  AMP/FP16
6.  `channels_last` if beneficial
7.  `torch.compile` only after numerical equivalence testing
8.  tune DataLoader workers
9.  reduce Python per-file overhead
10. vectorized save pipeline where practical

TensorRT should only be attempted if:

-   the final PyTorch path is already correct
-   conversion time is justified
-   output equivalence is tested
-   the repository remains reproducible

------------------------------------------------------------------------

# 6. AI/Codex Working Protocol

This is the most important process rule.

For every set, use this four-message cycle.

## Message A --- Plan only

> Review the current repository and the goal of Set N. Do not write
> code. Tell me exactly which files need to change, what interfaces you
> propose, what tests will prove correctness, and what risks you see.

Review the plan.

## Message B --- Implement only the agreed scope

> Implement only the approved Set N plan. Do not refactor unrelated
> files. Add tests. Show me the changed files and the commands I should
> run.

## Message C --- Run and diagnose

Paste command output.

> Here is the actual test/output log. Diagnose the failure. Do not
> change code yet. Identify the most likely root cause and the smallest
> fix.

## Message D --- Fix

> Apply only the smallest fix you proposed. Rerun the relevant tests. Do
> not proceed to Set N+1.

This prevents an AI agent from rewriting the entire repository after one
failure.

------------------------------------------------------------------------

# 7. Prompt Template for Experiment Runs

Use:

> We are running experiment `<ID>`. Freeze everything from the previous
> best run except `<single variable>`. Before editing, show the config
> diff. Then create the new config, run/train using the same split and
> seed policy, and write outputs to a new run directory. Never overwrite
> previous metrics or checkpoints. At the end, compare the new run
> against `<baseline ID>` using PSNR, SSIM, LPIPS, runtime and
> representative images.

------------------------------------------------------------------------

# 8. Prompt Template for Code Review

Use:

> Act as an external hackathon evaluator. Review the following files for
> reproducibility and evaluation risk. Focus on: hidden hard-coded
> paths, incorrect image ranges, incorrect metric data ranges, wrong
> grayscale LPIPS handling, filename mismatch, shape assumptions, output
> clipping, test leakage, end-to-end timing omissions, nondeterministic
> behavior, missing weights/configs, and commands that require source
> edits. Do not rewrite code. Return findings ordered by severity.

------------------------------------------------------------------------

# 9. Prompt Template for Final Submission Review

Use:

> Compare the final repository against the KLA Phase 1 requirements.
> Build a requirement-to-evidence matrix. For every requirement, point
> to the exact file/command/result that satisfies it. Mark anything
> unsupported as BLOCKER. Do not infer that a requirement is satisfied
> merely because the README claims it.

------------------------------------------------------------------------

# 10. Final Phase 1 Requirement-to-Repository Map

  Requirement             Repository evidence
  ----------------------- ----------------------------------
  Solution presentation   `presentation/...pdf`
  GitHub repository       repository URL
  Inference script        `inference.py`
  Training code           `train.py`, `src/engine/`
  Model weights/config    `weights/`, `configs/final.yaml`
  README                  `README.md`
  Dependencies            `requirements.txt`
  Results/samples         `results/`
  PSNR                    metric module + CSV
  SSIM                    metric module + CSV
  LPIPS                   metric module + CSV
  Baseline                bicubic + learned baseline
  Failure case            generated figure
  End-to-end runtime      `benchmark.py` + JSON
  External disclosures    `docs/external_resources.md`
  Reproducibility         clean-env log + tests

------------------------------------------------------------------------

# 11. Final Submission Checklist

## Model / data

-   [ ] Only official benchmark degradation types assumed
-   [ ] Unknown degradation order is not hard-coded
-   [ ] LR values outside `[0,1]` preserved intentionally
-   [ ] GT/output range treatment documented
-   [ ] scale factor inferred from official training pairs
-   [ ] final model works on variable spatial sizes if allowed

## Evaluation

-   [ ] PSNR correct
-   [ ] SSIM correct
-   [ ] LPIPS grayscale conversion correct
-   [ ] fixed validation split
-   [ ] no train/validation leakage
-   [ ] baseline included
-   [ ] final metrics machine-generated
-   [ ] failure case documented

## Runtime

-   [ ] H2D/D2H included
-   [ ] I/O included
-   [ ] saving included
-   [ ] batch size documented
-   [ ] precision documented
-   [ ] timing method documented

## Inference

-   [ ] input-dir argument
-   [ ] output-dir argument
-   [ ] all files processed
-   [ ] names preserved
-   [ ] correct shape
-   [ ] correct dtype
-   [ ] no NaN/Inf
-   [ ] no manual edits
-   [ ] checkpoint/config found automatically or via CLI

## GitHub

-   [ ] clean clone works
-   [ ] requirements install
-   [ ] tests pass
-   [ ] README commands tested
-   [ ] final config committed
-   [ ] final checkpoint available
-   [ ] external licences disclosed
-   [ ] final Git tag created

## Presentation

-   [ ] 6--7 slides maximum including title
-   [ ] organizer template used
-   [ ] instruction slide removed
-   [ ] metrics match repository
-   [ ] runtime matches repository
-   [ ] GitHub link works
-   [ ] demo link works if included
-   [ ] PDF reopened and visually checked

------------------------------------------------------------------------

# 12. Recommended Team Work Split

For a 3--4 person team, parallelize without creating conflicting code.

## Person A --- Data / evaluation

-   audit
-   split
-   metrics
-   error analysis
-   figures

## Person B --- Model / training

-   baseline
-   final model
-   losses
-   training configs

## Person C --- Inference / optimization

-   inference CLI
-   batching
-   benchmark
-   clean environment
-   packaging

## Person D --- Experiment management / PPT / documentation

-   run ledger
-   model card
-   README
-   presentation
-   requirement matrix

Merge through small pull requests or commits.

------------------------------------------------------------------------

# 13. "Do Not Do This" List

Do not:

-   clip NoisyLR at load time without an experiment
-   assume degradation order
-   train on hidden test inputs
-   report model-forward latency as end-to-end runtime
-   fabricate metric values for the PPT
-   tune against the hidden test set
-   calculate LPIPS directly on one-channel `[0,1]` tensors without
    correct adaptation
-   mix training and validation files
-   overwrite experiment directories
-   hard-code local Windows/Colab paths
-   depend on notebook cells
-   add a huge transformer simply because it sounds advanced
-   use an ensemble unless quality gain clearly justifies throughput
    cost
-   include unlicensed weights/data
-   claim unseen-degradation robustness beyond what KLA specifies
-   hide failure cases

------------------------------------------------------------------------

# 14. Minimum Viable Submission if Time Becomes Critical

If the deadline approaches, submit a **correct, reproducible, measured**
solution rather than an unfinished ambitious one.

Minimum strong package:

1.  dataset audit
2.  fixed validation split
3.  bicubic baseline
4.  trained compact residual or lightweight restoration candidate
5.  Charbonnier or Charbonnier+SSIM loss
6.  PSNR/SSIM/LPIPS
7.  tested inference CLI
8.  final weights/config
9.  README
10. end-to-end timing
11. representative success + failure images
12. compliant 7-slide PDF

A clean compact model can score better operationally than a large model
whose inference path fails.

------------------------------------------------------------------------

# 15. Suggested Work Order by Day

Given a short hackathon window, prioritize by dependency.

## Day 1

-   Sets 1--6
-   data audit
-   metrics
-   bicubic
-   one-pair overfit

## Day 2

-   Set 7
-   trained learned baseline
-   first complete result table

## Day 3

-   Sets 8--11
-   synthetic degradation
-   final architecture
-   loss/data ablations

## Day 4

-   Sets 12--17
-   throughput
-   inference hardening
-   clean environment
-   test dry run

## Day 5 / final day

-   Sets 18--21
-   figures
-   README
-   presentation
-   repository audit
-   PDF export
-   submission

Do not postpone the standalone inference script until the final hour.

------------------------------------------------------------------------

# 16. Definition of Done

The solution is done only when all of the following are true:

``` text
A clean clone can install.
A clean clone can run inference.
Every test input produces one correctly named output.
The final model checkpoint is available.
The validation protocol is frozen and reproducible.
PSNR, SSIM and LPIPS are reported.
End-to-end runtime is reported.
At least one baseline exists.
At least one failure case is documented.
Every presentation number is traceable to a result file.
The final PDF follows the organizer slide limit/template.
The submitted Git commit is tagged and unchanged after the final dry run.
```

------------------------------------------------------------------------

# Source Basis

This build plan is grounded in the uploaded:

-   `KLA_Problem Statement_Studen help document.pdf`
-   `KLA Problem Statement_explanation.pptx`
-   `Idea Submission Template_Hackathon 2026.pptx`
-   `Test_NoisyLR.zip`

It deliberately distinguishes between **confirmed KLA requirements** and
**recommended engineering choices**. Hyperparameters, model widths,
training epochs and synthetic-noise ranges must be measured/tuned from
the actual training data rather than treated as organizer-specified
facts.

------------------------------------------------------------------------

# B. Additional Explicit Gated Execution Playbook

# KLA SemiCon AI Hackathon 2026

# Gated Technical Execution Playbook

## How to use this document

Do not execute this entire document in one pass.

The process is intentionally gated.

Complete one phase.

Run the checks.

Record the evidence.

Only continue when the gate passes.

The goal is to prevent the team from spending days training a
sophisticated model on a broken data pipeline or an invalid validation
split.

------------------------------------------------------------------------

# Phase 0: Freeze the problem definition

## Objective

Create one internal technical contract that every team member follows.

## Tasks

Write a file:

``` text
docs/problem_contract.md
```

It must contain:

``` text
Input:
grayscale degraded image

Supported degraded sizes:
256 x 256
128 x 128

Target:
clean full resolution image

Supported target sizes:
512 x 512
256 x 256

Required degradation handling:
speckle noise
additive Gaussian noise
spatial resolution reduction

Combined degradation:
required

Generalization:
in distribution and out of distribution

Quality metrics:
SSIM
pSNR
LPIPS

Runtime:
must be measured

Deployment:
standalone evaluation script
```

## Exact questions

1.  What is the smallest input image size?
2.  What is the largest target image size?
3.  Is the input grayscale or RGB?
4.  Can one input contain more than one degradation?
5.  Can degraded pixel values exceed 1?
6.  Is ground truth always inside \[0, 1\]?
7.  Does the model need to increase spatial resolution?
8.  Is runtime measured?
9.  Is OOD data part of testing?
10. What exact directory arguments must the evaluator accept?

## Gate 0

Do not continue until every team member answers all ten questions
correctly without looking at the source.

------------------------------------------------------------------------

# Phase 1: Audit the actual training data

## Objective

Understand the data before designing the model.

Create:

``` text
scripts/audit_dataset.py
```

and:

``` text
reports/data_audit.csv
reports/data_summary.json
```

## Measure for every image

Record:

1.  Filename
2.  Height
3.  Width
4.  Channel count
5.  Minimum
6.  Maximum
7.  Mean
8.  Standard deviation
9.  1st percentile
10. 5th percentile
11. 50th percentile
12. 95th percentile
13. 99th percentile
14. Percentage below 0
15. Percentage above 1
16. Ground truth input pair identifier

## Exact questions

1.  Are all inputs grayscale?
2.  Are all targets grayscale?
3.  Which input dimensions occur?
4.  Which target dimensions occur?
5.  Is every input paired with exactly one target?
6.  Is every target paired with exactly one input?
7.  What percentage of degraded pixels are outside \[0, 1\]?
8.  What are the five largest absolute input intensities?
9.  Are there corrupted files?
10. Are there duplicate images?
11. Are there near duplicates?
12. Are there identifiable source groups?
13. Are some source groups much larger than others?
14. Is there a systematic intensity difference between source groups?

## Gate 1

Pass only when:

``` text
100 percent of files are readable
100 percent of pairs are mapped correctly
all image dimensions are known
intensity distributions are known
source grouping is understood
duplicates are identified
```

If any item fails, stop.

Do not train.

------------------------------------------------------------------------

# Phase 2: Build the validation split before training

## Objective

Prevent accidental data leakage.

Create:

``` text
data/
    train/
    validation_id/
    validation_ood/
    stress/
```

If source metadata exists, use source groups rather than random image
level splitting.

If source metadata does not exist, create the strongest defensible split
based on whatever structure is available and document the limitation.

## Validation sets

### Validation ID

Images similar to training.

Purpose:

Measure normal restoration accuracy.

### Validation OOD

Hold out an identifiable source or structure family.

Purpose:

Measure generalization.

### Stress set

Create controlled degradations with stronger but plausible degradation.

Purpose:

Measure robustness.

### Combined set

Explicitly ensure speckle and Gaussian noise can coexist with
downsampling.

Purpose:

Prevent the model from succeeding only when degradation types are
isolated.

## Exact questions

1.  Can the same original structure appear in train and validation?
2.  Can near duplicate images appear across splits?
3.  Can source information leak through filenames?
4.  Does validation OOD contain an unseen source?
5.  Does validation OOD have different structural content?
6.  Does the stress set contain stronger degradation?
7.  Does the combined set contain multiple degradation types?
8.  Are all evaluation images excluded from training?

## Gate 2

Do not proceed until the team can demonstrate that an image or near
duplicate cannot accidentally appear in both training and validation.

------------------------------------------------------------------------

# Phase 3: Establish the unavoidable baselines

## Objective

Find out how much of the task is solved without sophisticated AI.

Run at least:

### Baseline 1

Direct upsampling.

Use bicubic interpolation.

### Baseline 2

Simple convolutional restoration network.

### Baseline 3

A stronger residual or U Net style model.

Record:

``` text
Model
Parameters
Input size
Output size
SSIM
pSNR
LPIPS
Runtime
GPU memory
```

## Required experiment table

  Model               Parameters   SSIM   pSNR   LPIPS   Runtime   Memory
  ----------------- ------------ ------ ------ ------- --------- --------
  Bicubic                                                        
  CNN                                                            
  Strong baseline                                                

## Exact questions

1.  How much does bicubic improve over the degraded image?
2.  What does the CNN improve?
3.  Which degradation remains hardest?
4.  Does the strong baseline improve OOD performance?
5.  Does the strongest model justify its additional runtime?
6.  Does any model produce visible ringing?
7.  Does any model erase small structures?

## Gate 3

The team must have a numerical baseline table and at least five visual
comparisons before changing the architecture.

------------------------------------------------------------------------

# Phase 4: Choose the model geometry

## Objective

Build an architecture that naturally handles two times spatial
enlargement.

Do not blindly feed a 128 x 128 image into a model that expects 512 x
512.

The architecture must explicitly account for the scale factor.

A sensible starting design is:

``` text
Input 1 channel
       |
Shallow feature extraction
       |
Residual restoration blocks
       |
Feature refinement
       |
2x upsampling
       |
Output 1 channel
```

For a 128 x 128 input:

``` text
128 x 128
    |
    v
256 x 256 output
```

For a 256 x 256 input:

``` text
256 x 256
    |
    v
512 x 512 output
```

The implementation must determine the target scale from the input
dimensions or dataset pairing.

## Candidate architecture families

Test in this order:

1.  Compact residual CNN
2.  U Net restoration network
3.  lightweight restoration network
4.  Stronger attention or Transformer based restoration model

Do not start with the largest architecture.

## Exact questions

1.  Does the architecture support one channel?
2.  Does it support both scale cases?
3.  Does it preserve spatial alignment?
4.  Does the output exactly match the ground truth dimensions?
5.  Does the model introduce checkerboard artifacts?
6.  Does it fit comfortably in memory?
7.  Does runtime remain reasonable?
8.  Does a larger model produce a meaningful validation improvement?

## Gate 4

Select a primary model only after comparing at least two learned
architectures.

------------------------------------------------------------------------

# Phase 5: Handle input intensity correctly

## Objective

Decide how to preprocess degraded values without destroying useful
signal.

Run these experiments:

``` text
Experiment A:
raw degraded input

Experiment B:
input clipped to [0,1]

Experiment C:
input normalized using measured dataset statistics

Experiment D:
input normalized per image
```

Do not assume one is correct.

Measure all four.

## Record

``` text
SSIM
pSNR
LPIPS
OOD SSIM
OOD pSNR
OOD LPIPS
Runtime
```

## Exact questions

1.  Does clipping remove useful high intensity information?
2.  Does clipping improve or reduce OOD performance?
3.  Does per image normalization remove meaningful intensity
    information?
4.  Does dataset normalization remain stable across sources?
5.  Which method gives the best combined ID and OOD result?
6.  Does the chosen method preserve the relationship between input and
    ground truth?

## Gate 5

Select preprocessing from evidence.

Document the reason in:

``` text
docs/preprocessing_decision.md
```

------------------------------------------------------------------------

# Phase 6: Design the loss

## Objective

Prevent the model from optimizing only one notion of quality.

Start with:

``` text
L1
```

Then test:

``` text
L1 + structural loss
L1 + gradient loss
L1 + structural loss + gradient loss
```

Only add perceptual loss after measuring whether it helps.

## Required ablation

  ---------------------------------------------------------------------------------
  Loss          ID SSIM   ID pSNR  ID LPIPS  OOD SSIM  OOD pSNR OOD LPIPS   Runtime
  ----------- --------- --------- --------- --------- --------- --------- ---------
  L1                                                                      

  L1 +                                                                    
  structure                                                               

  L1 +                                                                    
  gradient                                                                

  Combined                                                                
  ---------------------------------------------------------------------------------

## Exact questions

1.  Does the additional loss improve SSIM?
2.  Does it improve pSNR?
3.  Does it improve LPIPS?
4.  Does it improve OOD performance?
5.  Does it create ringing?
6.  Does it sharpen noise?
7.  Does it create artificial edges?
8.  Does it reduce small defect fidelity?
9.  Is the improvement large enough to justify complexity?

## Gate 6

A loss term enters the final model only if the experiment demonstrates a
meaningful benefit.

------------------------------------------------------------------------

# Phase 7: Build degradation aware training

## Objective

Make the model robust to combined degradation.

Do not train only on the easiest cases.

Construct training categories:

``` text
Clean
Speckle
Gaussian
Downsampling
Speckle + Downsampling
Gaussian + Downsampling
Speckle + Gaussian
Speckle + Gaussian + Downsampling
```

The exact synthetic distributions must be based on what the supplied
data actually looks like.

Do not create arbitrary unrealistic noise simply to inflate training
diversity.

## Training schedule

Start with the actual paired dataset.

Then, if synthetic augmentation is justified:

``` text
Real paired data
+
controlled degradation augmentation
```

Do not let synthetic data completely dominate unless experiments show
that it helps.

## Exact questions

1.  Does the model see combined degradation during training?
2.  Does the training distribution resemble the hidden test problem?
3.  Are synthetic augmentations stronger than real training degradation?
4.  Does augmentation improve OOD?
5.  Does augmentation hurt ID performance?
6.  Are the synthetic transformations physically defensible?
7.  Can every augmentation parameter be explained?

## Gate 7

No augmentation becomes part of the final training recipe without an
ablation.

------------------------------------------------------------------------

# Phase 8: Attack the model with failure cases

## Objective

Find what breaks before the judges do.

Create a failure suite.

Include:

1.  Strong speckle
2.  Strong Gaussian noise
3.  Strong downsampling
4.  Combined noise
5.  High contrast edges
6.  Very thin structures
7.  Small isolated structures
8.  Dense structures
9.  Low contrast structures
10. Source shifted images
11. Extreme intensity values
12. Boundary regions

## For every failure save:

``` text
input.png
prediction.png
ground_truth.png
difference.png
metrics.json
```

## Exact questions

1.  What structure did the model remove?
2.  What structure did the model invent?
3.  Where is the largest error?
4.  Is the error caused by noise?
5.  Is the error caused by super resolution?
6.  Is the error caused by an intensity issue?
7.  Is the error source specific?
8.  Is the failure repeatable?
9.  Can the failure be fixed without hurting another category?

## Gate 8

Do not call the model robust until the failure suite has been reviewed.

------------------------------------------------------------------------

# Phase 9: Optimize runtime

## Objective

Reduce inference time without sacrificing restoration quality.

Measure runtime using the actual evaluation path.

Do not measure only the neural network forward pass if the evaluator
also includes image loading, preprocessing, postprocessing and writing.

Measure:

``` text
File loading
Preprocessing
Model inference
Postprocessing
File writing
Total
```

## Optimization order

1.  Remove unnecessary preprocessing
2.  Remove unnecessary model layers
3.  Reduce feature width if quality is preserved
4.  Use mixed precision if numerically safe
5.  Use inference mode
6.  Benchmark batch processing if evaluator behavior permits it
7.  Test compilation only if it remains reliable
8.  Consider ONNX only if the final environment supports it reliably

## Exact questions

1.  What is total time per image?
2.  What is model forward time?
3.  What percentage is file I/O?
4.  What percentage is preprocessing?
5.  Does mixed precision change metrics?
6.  Does model reduction change metrics?
7.  Does optimization alter output dimensions?
8.  Does optimization introduce numerical instability?

## Gate 9

Every speed optimization must pass:

``` text
Quality does not materially regress
AND
runtime improves
AND
the evaluator still runs
```

------------------------------------------------------------------------

# Phase 10: Build the exact evaluator

## Objective

Make the evaluator boring.

Boring is good.

Create:

``` text
inference/evaluate.py
```

It must accept:

``` text
input directory
output directory
```

The script must:

1.  Discover supported images
2.  Load the trained model
3.  Process every image
4.  Restore the image
5.  Save it to the output directory
6.  Preserve the expected filename relationship
7.  Exit cleanly

Do not require:

``` text
editing source code
changing a path inside Python
opening a notebook
downloading a checkpoint manually
changing configuration variables
```

## Gate 10

Clone the repository into a completely new environment.

Install only:

``` text
requirements.txt
```

Run the evaluator.

If it fails, the gate fails.

Repeat until it works.

------------------------------------------------------------------------

# Phase 11: Reproduce the final result

## Objective

Prove that the final model is reproducible.

Freeze:

``` text
model weights
training configuration
random seeds
requirements
model architecture
preprocessing
postprocessing
evaluation code
```

Record:

``` text
Git commit
Python version
PyTorch version
CUDA version
GPU
model parameter count
checkpoint size
```

## Exact questions

1.  Can another team member reproduce the validation result?
2.  Can a clean machine load the weights?
3.  Does the same input produce the same output within expected
    numerical tolerance?
4.  Is every dependency in requirements.txt?
5.  Does the evaluator work without internet access except for
    dependency installation if required?
6.  Is the checkpoint path fixed through robust configuration?
7.  Are there hidden assumptions about working directory?

## Gate 11

No final submission until the fresh environment test passes.

------------------------------------------------------------------------

# Phase 12: Create the evidence pack

## Required evidence

Create:

``` text
reports/
    baseline_table.csv
    loss_ablation.csv
    architecture_ablation.csv
    preprocessing_ablation.csv
    ood_results.csv
    runtime_results.csv
    failure_analysis.csv
```

Create visual panels:

``` text
input
prediction
ground truth
absolute error
```

For at least:

1.  Normal case
2.  Heavy noise case
3.  Strong downsampling case
4.  Combined degradation case
5.  OOD case
6.  Failure case

## Gate 12

Every performance number in the PPT must exist in one of the evidence
files.

------------------------------------------------------------------------

# Phase 13: Prepare the final model

## Freeze the final candidate

Record:

``` text
Architecture:
Loss:
Preprocessing:
Training data:
Augmentations:
Optimizer:
Learning rate:
Scheduler:
Epochs:
Batch size:
Checkpoint:
Parameter count:
Runtime:
```

Then do not change the model casually.

Any change requires rerunning the core validation suite.

------------------------------------------------------------------------

# Phase 14: Build the GitHub submission

## Repository checklist

``` text
README.md
requirements.txt
inference/evaluate.py
training/train.py
model/architecture.py
model/losses.py
weights/
outputs/
docs/
```

## README must answer

1.  What is the project?
2.  What problem does it solve?
3.  What environment is required?
4.  How are dependencies installed?
5.  Where are the weights?
6.  How is evaluation executed?
7.  What input directory is expected?
8.  What output directory is created?
9.  What output dimensions are produced?
10. How was the model trained?
11. What metrics were obtained?
12. What limitations remain?

## Gate 14

A person who has never spoken to the team must be able to run the
evaluator using only the README.

------------------------------------------------------------------------

# Phase 15: Submission hardening

## Perform this exact test

Create a fresh environment.

Clone the repository.

Install requirements.

Run:

``` text
python inference/evaluate.py INPUT_DIRECTORY OUTPUT_DIRECTORY
```

Then verify:

``` text
OUTPUT_DIRECTORY exists
number of outputs = number of valid inputs
every output is readable
every output has the expected dimensions
every output is grayscale
no output contains NaN
no output contains infinity
script exits with code 0
```

If any check fails, stop.

------------------------------------------------------------------------

# Phase 16: Final PPT evidence check

Before making slides, create a one page fact sheet containing only
measured facts.

Example:

``` text
ID SSIM:
ID pSNR:
ID LPIPS:

OOD SSIM:
OOD pSNR:
OOD LPIPS:

Average inference time:
Peak memory:
Parameter count:

Baseline SSIM:
Final SSIM:

Baseline pSNR:
Final pSNR:

Baseline LPIPS:
Final LPIPS:
```

Do not insert numbers into the PPT before this sheet is finalized.

------------------------------------------------------------------------

# Phase 17: Final decision gate

The final model is acceptable only if:

1.  It handles both resolution scales.
2.  It handles combined degradation.
3.  It does not rely on clipping blindly.
4.  It has ID validation.
5.  It has OOD validation.
6.  It has SSIM results.
7.  It has pSNR results.
8.  It has LPIPS results.
9.  It has runtime results.
10. It has failure analysis.
11. It has ablations.
12. The evaluator works from a clean environment.
13. The weights are accessible.
14. The repository is public.
15. The README is sufficient.
16. The PDF is compliant with the template.

Only after this gate should the team submit.

------------------------------------------------------------------------

# Recommended team split

## Person 1: Data

Own:

``` text
data audit
validation split
augmentation
OOD analysis
```

## Person 2: Model

Own:

``` text
architecture
training
loss
ablations
```

## Person 3: Evaluation

Own:

``` text
metrics
failure analysis
runtime
benchmarking
```

## Person 4: Submission

Own:

``` text
GitHub
README
PPT
references
final compliance
```

Every person should still understand the complete pipeline.

------------------------------------------------------------------------

# The golden rule

Never move from:

``` text
idea
```

directly to:

``` text
final model
```

Always move through:

``` text
hypothesis
experiment
measurement
comparison
decision
documentation
```

That is how the team turns a plausible hackathon project into an
evidence backed engineering solution.

## CURRENT SOURCE BASELINE — AUGUST 2026

This file has been reconciled against the newest participant materials supplied for the KLA problem statement.

Current primary sources:

1. `KLA_Problem Statement_Studen help document.pdf`
2. `Idea Submission Template_Hackathon 2026.pptx`
3. `KLA Problem Statement_explanation(1).pptx`
4. `q&A session transcript.txt`
5. `problemstatement.txt`
6. `combinedreport.txt`
7. The latest participant notice supplied with the dataset link and Phase 1 deadline

Older MD files are reference material only and do not override these sources.

### Current confirmed benchmark contract

The official benchmark degradation mechanisms are:

1. Additive Gaussian noise
2. Multiplicative speckle noise
3. Spatial downsampling

The order of those mechanisms is not disclosed and may vary. The model does not have to identify the order.

The latest help document says these are the only benchmark degradation mechanisms. Earlier educational material demonstrates blur when explaining image degradation in general, but blur is not treated as a fourth benchmark corruption.

### Current data contract

Training data consists of paired GT and NoisyLR images.

GT is normalized to `[0,1]`.

NoisyLR may extend slightly outside `[0,1]`. This is intentional.

The actual dataset controls the exact dimensions and file encoding. Session guidance indicates approximately 256 x 256 and 512 x 512 evaluation images, with downsampling factors discussed from about 1.5x to 4x.

The hidden test set contains degraded inputs only. KLA retains the clean GT for scoring.

The test distribution contains both familiar and unfamiliar image content. The same three degradation mechanisms remain in the OOD set, with similar noise sampling ranges. The primary distribution shift is image content.

Phase 1 session guidance says the supplied Phase 1 data contains normal natural imagery and no physical manufacturing defects.

### Current development permissions

Any suitable restoration architecture is allowed.

Permitted directions include CNNs, transformers, algorithm unrolling, published architectures and justified custom or hybrid designs.

Public external datasets and pretrained weights may be used when their licenses permit competition use.

Every external dataset or model used in the final solution must be disclosed with its name, link, license and relevant paper or model/dataset card.

Synthetic degraded pairs may be generated from provided GT images.

Frequency-domain methods are allowed but not mandatory.

There is no fixed parameter-count ceiling, but unnecessarily large models can lose throughput.

### Current evaluation

KLA evaluates three high-level axes:

1. Restoration quality
2. End-to-end throughput
3. Training and compute hygiene

Restoration quality uses a fixed internal weighted combination of:

1. PSNR
2. SSIM
3. LPIPS

The exact weights are confidential.

Do not invent official metric or axis weightages.

End-to-end runtime includes:

1. Disk reading
2. Preprocessing
3. CPU to GPU transfer
4. Model execution
5. GPU to CPU transfer
6. Postprocessing
7. Disk writing

Benchmarking uses an NVIDIA H100.

Batch processing is strongly preferred. If a model cannot fit useful batches, single-image execution remains a fallback.

KLA does not apply output clipping or normalization before scoring. Final output range and encoding must therefore be handled intentionally inside the solution according to the official file/evaluator contract.

Metrics are computed on the full-resolution restored image, not on zoomed crops.

### Current reproducibility requirements

The inference script must be standalone, accept input and output directories, process the degraded images, save restored outputs, include required weights/config/dependencies and require no manual source-code or notebook-cell edits.

Training code must reproduce the submitted checkpoint.

The repository must be accessible and contain a clear structure.

### Current competition timeline

Phase 1 submission deadline: 16 August 2026.

The supplied consolidated KLA material describes a later shortlist of about 15 teams per problem statement followed by a Grand Finale of 5 teams per problem statement. This later-stage information is useful context but does not change Phase 1 deliverables.

### Current presentation conflict to remember

The current organizer PPT template says:

* remove the instruction slide
* use the supplied template
* keep the final deck to six or seven slides including the title
* avoid paragraphs and prefer points, diagrams, infographics and pictures
* save the final submission as PDF
* do not submit PPT, Word or another format according to the template instruction

The current KLA help document separately contains a long 12-section recommended solution presentation structure. Therefore the content categories must be compressed into the six or seven slide limit rather than treated as twelve final slides.

The KLA help document's Phase 1 deliverables table also calls the solution artifact PPT/PPTX, while the current template explicitly says the final portal upload should be PDF. Treat this as a format wording conflict and re-check the portal immediately before upload. Authoring should use the provided PPT template; final portal format should follow the latest portal/template instruction.


## Updated source-aligned execution gates

### Gate A: Source freeze

Before coding, verify:

[ ] latest KLA help document read

[ ] current submission template read

[ ] latest Q&A read

[ ] original KLA presentation reviewed

[ ] current participant notice recorded

[ ] source conflicts logged

### Gate B: Data first

Before model selection:

1. inspect real GT and NoisyLR
2. compute range statistics
3. verify pair alignment
4. verify dimensions
5. visualize histograms
6. inspect out-of-range NoisyLR values
7. freeze train/validation split

### Gate C: Basic pipeline

Build:

1. loader
2. tiny model
3. simple regression loss
4. metrics
5. checkpoint
6. inference

Then overfit one or two samples.

### Gate D: Baseline

At least one baseline must be compared with the final method.

Recommended baselines:

* nearest
* bilinear
* bicubic
* tiny CNN

### Gate E: Component optimization

Test one component at a time:

1. preprocessing
2. augmentation
3. model capacity
4. loss
5. frequency contribution
6. pretrained initialization
7. runtime

### Gate F: OOD

Evaluate content shift.

Do not invent a fourth degradation.

### Gate G: Runtime

Profile complete end-to-end execution on NVIDIA GPU.

Measure:

disk read

preprocessing

host-to-device

model

device-to-host

postprocessing

disk write

### Gate H: Reproducibility

Fresh environment:

```text
clone
install
weights
run inference
verify outputs
```

No code edits.

### Gate I: Phase 1 package

The final package must include:

* presentation
* GitHub
* inference
* training code
* weights/config
* README
* dependencies
* results/output examples
* disclosures

### Gate J: Slide compression

The technical guidance suggests many presentation content sections, but the current organizer template caps the final deck at six or seven slides.

Merge content, do not create a twelve-slide submission.

## Development roadmap aligned to KLA mentor advice

### Stage 1: Visual data inspection

* download official data
* plot GT/NoisyLR pairs
* inspect histograms
* inspect ranges

### Stage 2: Simple baseline

* basic loader
* simple model
* simple L1/L2 loss
* baseline metrics

### Stage 3: Overfit 1–2 images

If this fails, fix the pipeline.

### Stage 4: Component-by-component optimization

* preprocessing
* augmentation
* architecture
* capacity
* losses
* frequency

### Stage 5: Systems tuning

* batching
* data pipeline
* GPU utilization
* image I/O
* output formatting
* runtime

## Time-critical minimum path

If time becomes constrained:

1. Get official data loader working.
2. Build bicubic baseline.
3. Build small trainable model.
4. Pass one/two-image overfit.
5. Establish clean validation split.
6. Implement PSNR, SSIM, LPIPS.
7. Choose one strong compact architecture.
8. Add one justified composite loss.
9. Add validated augmentation.
10. Build clean inference script.
11. Test end-to-end.
12. Build the submission deck.


## Current source guard for legacy experiment scaffolding

Some older gated experiment examples in this file contain concrete sample hyperparameters, model widths, patch sizes, batch sizes or output policies.

Those values are scaffolding only.

They are not KLA-provided values and must not be treated as official settings.
The current official dataset, measured validation behavior and benchmark requirements determine the actual values to use.
