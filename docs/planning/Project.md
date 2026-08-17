# Project.md

# SEMICON India Hackathon 2026 --- KLA

# AI-Based Restoration of Degraded Images for Semiconductor Inspection

> **Authoritative project scope:** This six-file project system is
> exclusively for the KLA problem statement **AI-Based Restoration of
> Degraded Images for Semiconductor Inspection**. Any material
> concerning other hackathon problem statements is intentionally
> excluded.

## 0. How this file must be used

`Project.md` is the project requirements and success-definition
document. It answers **what are we building, why are we building it, who
is it for, what exactly must it do, what constraints are known, what
must be submitted, and what does a strong solution look like?**

An AI coding agent must read this file before changing architecture or
implementation. It must not treat recommendations as official
requirements. Where this document says **official/source-derived**, the
claim is grounded in the supplied KLA materials. Where it says
**recommended strategy**, it is an engineering recommendation and may be
changed only through evidence and an entry in `memory.md`.

## 1. Problem statement

The task is to learn the transformation from a degraded semiconductor
inspection image to its clean ground-truth image. The supplied KLA
presentation describes the task as image restoration: given degraded
images, obtain clean images by learning the transform. The documented
degradation mechanisms are **speckle noise, down-sampling, and additive
Gaussian noise**. The presentation explicitly warns that real-world
degradation can occur in different orders, so the implementation must
not assume a fixed degradation sequence.

The training data is paired: degraded input and corresponding ground
truth. Test data can be similar to or dissimilar from the training
distribution. The challenge is therefore not only reconstruction quality
on familiar data; it is also robust generalization.

The source presentation explicitly identifies four areas to emphasize
while building the model:

1.  Data
2.  AI model
3.  Loss
4.  Compute

It also states that evaluation is not just about leaderboard statistics
and that computation and training hygiene are critical.

## 2. Why the problem is difficult

This is an inverse problem. Degradation removes or corrupts information.
A restoration model must infer plausible clean structure from incomplete
and noisy observations. In semiconductor inspection, small structures
and edges matter, so visually pleasing smoothing is not enough. A model
can be numerically or perceptually attractive while still inventing a
line, removing a real feature, introducing ringing, or erasing a defect.

The target therefore is **faithful restoration**, not generic image
enhancement.

## 3. Exact task contract

### Input

-   Grayscale degraded image.
-   Supported degraded resolutions documented in the supplied materials
    are `256×256` and `128×128`.
-   Degraded images may contain intensity values outside `[0,1]`. This
    is explicitly described as a feature, not a bug.

### Ground truth

-   Grayscale clean image.
-   Supported ground-truth resolutions are `512×512` and `256×256`.
-   Ground truth is always within `[0,1]`.

### Resolution relationship

  Degraded input   Ground truth / required output     Scale
  ---------------- -------------------------------- -------
  `128×128`        `256×256`                             2×
  `256×256`        `512×512`                             2×

### Degradation contract

The model must handle:

-   speckle noise
-   additive Gaussian noise
-   spatial resolution reduction / down-sampling
-   combinations of the above

Do not implement three independent models and assume that their isolated
performance proves the combined task is solved.

## 4. Data rules that are part of the project

The degraded input must not be silently clipped to `[0,1]` merely
because many image libraries expect normalized images. First measure the
actual distribution. The data pipeline must intentionally decide how
values are normalized, preserved, or clipped, and that decision must be
validated experimentally.

Before serious training, the team must know:

1.  file format
2.  channel count
3.  input dimensions
4.  target dimensions
5.  dtype
6.  pairing rule
7.  minimum and maximum input values
8.  percentage of values below zero
9.  percentage above one
10. source/group distribution where available
11. duplicate or near-duplicate risk
12. degradation characteristics

The implementation should infer inspectable facts from the official data
rather than hard-code assumptions.

## 5. Target users

### Primary user: KLA evaluator

The evaluator needs a public repository and a standalone
inference/evaluation path that can be executed without editing source
code. The repository must make the solution reproducible and
benchmarkable.

### Secondary user: technical judge

The judge needs to understand the problem, why the chosen model exists,
what evidence supports the choices, whether the model generalizes, how
fast it runs, and whether the implementation is credible.

### Development user: the team

The team needs a controlled workflow that prevents half-built
architecture, invalid metrics, data leakage, and undocumented
experiments.

## 6. Functional requirements

The final system must:

1.  Load the official degraded images.
2.  Pair training images with the correct ground truth.
3.  Support both documented resolution scales.
4.  Preserve or intentionally transform degraded values according to an
    experimentally validated preprocessing policy.
5.  Restore clean full-resolution images.
6.  Handle combined degradation.
7.  Produce outputs in the exact expected geometry and format.
8.  Evaluate using PSNR/pSNR, SSIM and LPIPS as required by the supplied
    materials.
9.  Measure inference speed.
10. Support in-distribution and OOD validation.
11. Provide a reproducible training path.
12. Provide a standalone evaluation path.

## 7. Quality requirements

A strong solution should optimize the joint objective:

``` text
restoration fidelity
+ structural preservation
+ OOD robustness
+ combined-degradation robustness
+ computational efficiency
+ reproducibility
```

There is no invented official percentage weighting in this document. If
the organizers publish a numeric rubric, it must be recorded in
`memory.md` and treated as authoritative.

## 8. Submission requirements

The supplied description requires a public GitHub repository containing:

-   `README.md`
-   standalone evaluation script
-   training script
-   trained weights
-   restored outputs
-   `requirements.txt`

The evaluator must accept a test-image directory and an output directory
and must work as supplied. The supplied description states that KLA will
benchmark the evaluation script on an H100.

The supplied presentation/template requires a PDF using the
organizer-provided template and preserving the core idea-detail
pointers. The template instruction says 6--7 slides including the title
slide. Another supplied description states a maximum of 8--9 slides.
This is a source conflict and must not be silently invented away. Until
clarified, use the stricter template constraint of 6--7 slides and
record any official clarification in `memory.md`.

## 9. What the presentation must communicate

The final deck must compress the following technical story into the
allowed slide count:

-   semiconductor inspection context
-   paired GT ↔ degraded restoration task
-   speckle, Gaussian and downsampling degradation
-   dataset observations
-   input/output value-range handling
-   restoration pipeline
-   preprocessing and augmentation
-   architecture and design rationale
-   loss and training setup
-   baseline comparison
-   PSNR/pSNR, SSIM and LPIPS
-   runtime and timing method
-   visual results
-   failure cases and limitations
-   innovation/uniqueness
-   impact and feasibility
-   external-resource disclosure
-   GitHub and reproduction commands
-   research references

## 10. Winning philosophy

The strongest strategy is not automatically the largest model. The
recommended path is evidence-driven:

``` text
data correctness
      ↓
metric correctness
      ↓
trivial baseline
      ↓
tiny learning test
      ↓
compact strong baseline
      ↓
degradation calibration
      ↓
final candidate architecture
      ↓
loss ablation
      ↓
augmentation ablation
      ↓
model-size / throughput search
      ↓
output-range policy
      ↓
error analysis
      ↓
final inference hardening
      ↓
clean-environment test
      ↓
submission package
```

The important principle is **do not skip evidence-producing intermediate
stages**. A sophisticated model built on a broken dataset split or
incorrect metric implementation is worse than a simple model with a
trustworthy pipeline.

## 11. Failure modes the project must actively prevent

### Data leakage

If structurally related or duplicate samples enter both training and
validation, the validation score can become misleading.

### Blind input clipping

Clipping NoisyLR values before understanding the distribution may remove
information.

### Overfitting to ID data

Excellent ID metrics can hide poor performance on dissimilar test
sources.

### Hallucination

Aggressive super-resolution or perceptual optimization can invent
structures.

### Metric mistakes

LPIPS requires deliberate grayscale handling and normalization. PSNR
must use the correct data range. SSIM must be computed correctly for
grayscale.

### Runtime blindness

A model that is excellent but excessively slow may be inferior to a
slightly less accurate model with much better end-to-end throughput.

### Submission fragility

A correct model that cannot be executed from a clean environment is not
a finished hackathon solution.

## 12. Definition of done

The project is done only when:

-   data audit is complete
-   pairing is verified
-   validation leakage is addressed
-   metrics pass unit tests
-   baseline exists
-   final model is selected through experiments
-   combined degradation is tested
-   OOD behavior is measured
-   failure cases are analyzed
-   runtime is measured end to end
-   evaluator works in a clean environment
-   weights are packaged
-   README is sufficient for an outsider
-   presentation numbers are traceable to experiments
-   final PDF is template compliant
-   `memory.md` records the final state

## 13. Source-derived versus recommended content

### Source-derived

The KLA materials establish the task, degradation types, paired data,
grayscale/value behavior, resolution relationships, ID/OOD testing,
evaluation metrics, speed considerations, compute/training hygiene
emphasis, and submission requirements.

### Recommended strategy

The detailed baseline ladder, specific architecture candidates, ablation
sequence, repository organization, experiment gates and engineering
practices in the companion files are recommended execution strategy.
They are not presented as official organizer scoring weights.

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


## Reconciled technical interpretation

### Image degradation as an inverse problem

Let `X` be the clean GT and `Y` the observed NoisyLR image.

`Y = f(X)`

The restoration system estimates:

`X_hat = F(Y)`

The inverse is not unique because information is removed by noise and downsampling.

The engineering target is therefore a good regularized inverse estimate rather than an exact physical inversion.

### Gaussian noise

A simple additive model is:

`Y = X + N_G`

with zero-mean Gaussian noise in the conceptual model.

The KLA presentation illustrates increasing sigma as an increasing corruption severity.

### Speckle noise

A simple multiplicative model is:

`Y = X * N_S`

Speckle therefore interacts with local signal intensity.

### Downsampling

The session materials illustrate examples such as:

`512 x 512 -> 256 x 256`

and discuss factors from approximately 1.5x through 4x.

The exact official dataset dimensions remain authoritative.

### Combined corruption

The benchmark may contain combinations of the three mechanisms.

The order may vary.

A model can therefore be:

* implicit one-shot restoration
* staged restoration
* degradation-aware restoration

Explicit sequence prediction is optional.

## The four engineering axes

KLA's technical guidance can be organized into four practical R&D axes:

1. Data and augmentation
2. Model architecture
3. Loss and objective design
4. Compute and systems engineering

This is a development framework, not an official scoring formula.

### Data and augmentation

KLA encourages:

* image augmentation
* synthetic degradation from GT
* external public image datasets
* domain adaptation
* pretraining

The primary rationale is robustness and OOD generalization.

### Model architecture

KLA permits:

* CNN
* transformer
* algorithm unrolling
* published architecture
* custom/hybrid architecture

Useful priors can include:

* edge preservation
* multiscale features
* frequency representations
* degradation-aware blocks

### Loss design

Candidate loss families include:

* L1
* L2
* Charbonnier
* differentiable SSIM
* frequency-domain losses
* perceptual losses

Binary cross entropy is not appropriate as the default loss for this continuous image-regression task.

The final loss must be selected experimentally.

### Compute and systems

Compute is relevant during both training and inference.

Useful directions include:

* efficient data loading
* pinned memory
* non-blocking transfers where safe
* mixed precision
* `torch.compile` or equivalent framework acceleration where stable
* keeping expensive loss calculations on GPU
* reducing unnecessary conversions
* efficient image encoding and writing
* batch-size tuning

Do not optimize the forward pass while ignoring I/O.

## Critical development clarification

The supplied Phase 1 data is normal natural imagery.

The semiconductor context explains why image restoration matters, but Phase 1 does not require defect-aware learning.

Therefore the model should optimize restoration quality and generalization rather than a defect-preservation objective.


## Essential non-negotiable rules

1. Treat the latest KLA participant help document and current organizer template as primary sources.
2. Do not treat blur as a fourth benchmark degradation.
3. Do not assume a fixed degradation order.
4. Do not blindly clip NoisyLR before measuring or deciding how it should be handled.
5. Do not use hidden-test outputs as an iterative training signal unless KLA explicitly permits it.
6. Do not invent KLA scoring weights or a target latency threshold.
7. Do not report unmeasured results.
8. Do not use unlicensed external data or model weights.
9. Do not depend on notebook state for final inference.
10. Do not hard-code local Windows, Colab or Kaggle paths into the final submission.
11. Record every important experiment with its configuration and commit.
12. The final inference package must work without manual source edits.
13. The final presentation must stay inside the current six or seven slide limit including the title and use the supplied template.
14. At upload time, re-check the portal for the latest file format, naming convention and cut-off time.
