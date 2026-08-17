# Architecture.md

# SEMICON India Hackathon 2026 --- KLA

# System, Repository, Data, Model, Evaluation and Deployment Architecture

## 0. Architectural contract

The architecture is intentionally separated into:

``` text
data / I/O
    ↓
validation / metrics
    ↓
models / losses
    ↓
training engine
    ↓
checkpoints
    ↓
inference
    ↓
benchmark / artifacts
```

The architecture must support two modes:

1.  **Research mode:** training, validation, ablation and diagnostics.
2.  **Submission mode:** deterministic inference/evaluation with the
    smallest possible dependency surface.

The final evaluator must not depend on a notebook, interactive session,
hidden local paths or manual source modification.

## 1. Target repository architecture

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
│   ├── residual_candidate.yaml
│   ├── lightweight_candidate.yaml
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
│   │   ├── restoration_blocks.py
│   │   └── restoration_candidate.py
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

## 2. Module responsibilities

### `src/data/io.py`

Owns raw file reading/writing and format-specific handling. It must not
contain model logic.

Responsibilities:

-   read supported input
-   preserve dtype until intentional conversion
-   expose shape/channel information
-   write outputs deterministically
-   make range conversions explicit

### `src/data/dataset.py`

Owns pairing and tensor conversion for training. It must validate the
input/GT relationship rather than silently repairing mismatches.

### `src/data/split.py`

Owns deterministic train/validation partitioning and source-aware
grouping where metadata permits. The split must be serialized so
experiments can reproduce it.

### `src/data/augment.py`

Owns paired-safe geometric augmentation and experimentally justified
degradation augmentation.

### `src/data/synthetic_degradation.py`

Owns synthetic degradation only after the real paired data has been
analyzed. It must expose parameters explicitly so calibration can be
measured.

### `src/models/`

Contains the baseline and candidate restoration architectures. The final
model must not be entangled with dataset loading.

### `src/losses/`

Contains modular loss terms so ablations can change one loss component
without rewriting training.

### `src/metrics/`

Contains PSNR, SSIM, LPIPS and aggregate reporting. Metric preprocessing
must be explicit and tested.

### `src/engine/`

Owns training, validation and checkpoint lifecycle.

### `src/utils/`

Owns reproducibility, configuration, device selection, logging and
timing.

## 3. Data contract

Conceptually each training record is:

``` text
{
  degraded: image,
  ground_truth: image,
  metadata: optional source/identity information
}
```

Validation requirements:

-   input channel count must be 1
-   GT channel count must be 1
-   input and GT must be paired correctly
-   output scale must be 2× for the documented cases
-   invalid dimensions must be surfaced, not silently changed

## 4. Tensor contract

Training tensors should use:

``` text
input  = [B, 1, H, W]
target = [B, 1, 2H, 2W]
output = [B, 1, 2H, 2W]
```

The implementation must not assume a single spatial size if the official
data contains both documented cases.

## 5. End-to-end training architecture

``` text
Paired data
   ↓
Dataset audit / pairing validation
   ↓
Train/ID/OOD split
   ↓
Preprocessing
   ↓
Optional calibrated degradation augmentation
   ↓
Restoration model
   ↓
Loss computation
   ↓
Backpropagation
   ↓
Optimizer / scheduler
   ↓
Checkpoint
   ↓
Validation
   ↓
Experiment artifact
```

## 6. End-to-end inference architecture

``` text
input_dir
   ↓
file discovery
   ↓
input validation
   ↓
preprocessing
   ↓
GPU transfer
   ↓
model inference
   ↓
postprocessing / output-range policy
   ↓
CPU transfer
   ↓
file save
   ↓
output_dir
```

The inference path must be the same path that is benchmarked. Do not
report only forward-pass timing if the evaluator experiences file I/O,
preprocessing and saving.

## 7. Evaluation architecture

``` text
GT + prediction
   ├── PSNR / pSNR
   ├── SSIM
   └── LPIPS

Input directory + evaluator
   └── end-to-end runtime
```

LPIPS requires deliberate grayscale handling. The implementation
described in the original blueprint replicates the grayscale channel to
three channels and maps `[0,1]` to `[-1,1]` before LPIPS. This must be
tested rather than assumed.

PSNR should use `data_range=1.0` when the compared restoration/GT values
are in `[0,1]`. SSIM must be configured for grayscale.

## 8. Model architecture progression

The recommended progression is:

``` text
Bicubic
   ↓
Tiny residual network
   ↓
compact residual / super-resolution candidate
   ↓
lightweight restoration candidate
   ↓
Model-size / throughput Pareto selection
```

Do not start with the most complex model. The progression provides
evidence for the value of learned restoration and creates a baseline for
every subsequent improvement.

## 9. Loss architecture

A modular total loss may be represented as:

``` text
L_total = λ_pixel L_pixel
        + λ_struct L_struct
        + λ_grad L_grad
        + optional λ_perceptual L_perceptual
```

The exact final formula is not fixed in advance. Each additional term
requires an ablation.

## 10. Checkpoint architecture

A checkpoint should preserve enough information to reconstruct the exact
inference configuration. Recommended contents:

``` text
model_state_dict
optimizer_state_dict (training checkpoints)
scheduler_state_dict (if used)
epoch
validation metrics
configuration
random seed
git commit if available
```

The final inference artifact may contain only what is needed for
submission, but the mapping between final weights and final
configuration must remain documented.

## 11. Configuration architecture

Use one explicit final configuration:

``` yaml
seed:
model:
input_channels: 1
scale: 2
loss:
optimizer:
learning_rate:
batch_size:
epochs:
augmentation:
normalization:
precision:
```

Do not allow different hard-coded values in training and inference.

## 12. Test architecture

Minimum unit/integration coverage:

-   `test_io.py` --- file loading and saving
-   `test_pairing.py` --- input/GT mapping
-   `test_shapes.py` --- supported dimensions and output scale
-   `test_metrics.py` --- metric sanity
-   `test_model_forward.py` --- model tensor contract
-   `test_inference_contract.py` --- CLI/output behavior

Metric sanity must include an identical-image case and a perturbed-image
case.

## 13. Artifact architecture

Every experiment should leave:

``` text
config
checkpoint
metrics CSV/JSON
training log
representative images
experiment note
```

The presentation must be assembled from these artifacts, not manually
remembered numbers.

## 14. Runtime architecture

Measure:

1.  disk read
2.  preprocessing
3.  host-to-device transfer
4.  model execution
5.  device-to-host transfer
6.  postprocessing
7.  disk save
8.  total elapsed time

Report both total time and ms/image when meaningful. Batch size, GPU,
warm-up policy and timing method must be recorded.

## 15. Clean-environment architecture

The repository must be testable from a new environment. The clean test
must verify:

-   dependency installation
-   model import
-   checkpoint loading
-   evaluator invocation
-   complete directory processing
-   expected output dimensions
-   no NaN/Inf
-   successful exit

## 16. Architecture decisions must be evidence-driven

Every major architectural component must answer:

1.  What problem does it solve?
2.  What experiment demonstrates the benefit?
3.  What is the runtime/memory cost?
4.  Does the benefit survive OOD testing?
5.  Can the component be removed without breaking the evaluator?

------------------------------------------------------------------------

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


## Current architecture requirements

### Input interface

Conceptually:

`B x 1 x H x W`

where the actual H and W come from the official dataset.

### Output interface

Conceptually:

`B x 1 x H_GT x W_GT`

with the corresponding GT resolution.

### Inference CLI

Recommended interface:

```bash
python inference.py --input_dir /path/to/noisy_lr_images --output_dir /path/to/restored_images
```

The final evaluator contract requires the two directory arguments.

### Output handling

Do not leave output range handling ambiguous.

The final inference pipeline must perform the appropriate:

* range conversion
* clipping or scaling when required
* dtype conversion
* file encoding
* naming

according to the official dataset/evaluator contract.

### Runtime-aware architecture

The benchmark counts the entire pipeline.

Therefore architecture design must consider:

* model compute
* memory movement
* preprocessing
* batching
* postprocessing
* writing

### Tiling

Tiling is permitted when useful.

If used, the time to process all tiles counts toward inference runtime.

The session material says evaluation images are not extremely large, commonly around 256 x 256 or 512 x 512, so tiling should be justified rather than assumed.

### Capacity strategy

KLA's mentor recommendation is:

1. Establish an accuracy ceiling with a sufficiently capable model.
2. Locate the point of diminishing returns.
3. Reduce cost using pruning, regularization, distillation or architectural simplification.
4. Re-benchmark the complete pipeline.

This is usually safer than starting from an arbitrary tiny network and never knowing the attainable quality ceiling.

### Pretrained models

Permitted and encouraged where useful.

Candidate sources discussed by the organizers include:

* Hugging Face
* PyTorch model ecosystem
* timm
* TensorFlow Model Zoo
* Torch Hub

Record licenses and citations.

### Architecture specialization

There is no requirement to explicitly encode degradation type or sequence.

However, degradation-specific architectural priors are explicitly a valid design direction.

The choice should be driven by experiments.

### Frequency branch

A frequency-domain branch or frequency loss is a legitimate experiment.

It is not mandatory.

A useful comparison is:

`spatial-only`

versus

`spatial + frequency`.

## Architecture acceptance gate

A candidate architecture should not become final until it has:

1. Correct tensor shape behavior
2. Successful one/two-pair overfit
3. Validation metrics
4. OOD evidence
5. Combined degradation evidence
6. Runtime evidence
7. Clean inference execution
