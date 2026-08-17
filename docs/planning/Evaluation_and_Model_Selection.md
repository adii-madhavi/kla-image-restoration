# Evaluation_and_Model_Selection.md
# KLA Restoration Evaluation, Ablation, Data Forensics and Winning Strategy


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



## 1. Official evaluation pillars

### Pillar A: Restoration quality

Fixed internal combination:

* PSNR
* SSIM
* LPIPS

Both ID and OOD content matter.

### Pillar B: End-to-end throughput

Common H100.

Full pipeline timing.

### Pillar C: Training and compute hygiene

Reproducibility.

Experiment discipline.

Code quality.

Efficient data pipeline.

## 2. The model must beat a baseline

Always keep at least one simple baseline.

Recommended:

1. Bicubic
2. Small CNN

The final model should be compared against the baseline on both quality and runtime.

## 3. Metric interpretation

### PSNR

Pixel-fidelity diagnostic.

### SSIM

Structural diagnostic.

### LPIPS

Perceptual diagnostic.

Do not optimize one metric blindly because the hidden score combines all three.

## 4. Full-resolution rule

Official scoring is on the final full-resolution restored image.

Zoomed crops are for qualitative inspection only.

## 5. OOD

Report separately:

* ID PSNR/SSIM/LPIPS
* OOD PSNR/SSIM/LPIPS

Do not call stronger corruption OOD when the primary shift is image content.

## 6. Combined degradation

Use controlled diagnostic sets for:

1. Gaussian
2. Speckle
3. Downsampling
4. Gaussian + Speckle
5. Gaussian + Downsampling
6. Speckle + Downsampling
7. All three

The hidden test itself does not require unseen degradation types.

## 7. Degradation-order experiment

Synthetic diagnostic sequences may be used to verify that the restoration system does not rely on a single fixed order.

Do not add blur to the default benchmark simulation.

## 8. Data forensics

Before major training, audit:

* shape
* dtype
* channel count
* range
* histogram
* pair alignment
* duplicates
* source grouping

Keep a dataset manifest.

## 9. Range policy experiment

Compare:

A. raw or minimally transformed NoisyLR

B. controlled normalization

C. alternative robust normalization

Do not clip blindly.

## 10. Loss ablation

At minimum compare:

1. L1
2. L2 or Charbonnier
3. L1 + SSIM
4. L1 + frequency
5. another justified composite

Keep architecture fixed during the core loss study.

## 11. Frequency-domain study

KLA explicitly identifies frequency-domain training as a valid direction.

Compare spatial-only versus spatial + frequency.

Keep it only if evidence supports it.

## 12. Pretrained model study

Compare:

* random initialization
* public pretrained initialization

Record:

* convergence
* quality
* OOD
* runtime
* licensing

## 13. Capacity study

KLA mentor advice:

1. Find a quality ceiling.
2. Locate diminishing returns.
3. Compress or simplify.
4. Re-test the full pipeline.

## 14. Augmentation study

Compare:

* no augmentation
* standard augmentation
* calibrated synthetic degradation
* stronger augmentation

The explicit Q&A recommends augmentation as a strong path to OOD robustness.

## 15. Failure analysis

Create at least:

1. best case
2. typical case
3. hard noisy case
4. OOD case
5. failure case

Classify failures such as:

* residual Gaussian noise
* residual speckle
* over-smoothing
* ringing
* missing detail
* invented detail
* OOD weakness

## 16. Runtime experiment

Record:

* batch size
* total image count
* disk read
* preprocessing
* transfers
* model
* postprocess
* output write
* total

## 17. Training hygiene

Record every meaningful experiment.

Recommended fields:

```text
run_id
git_commit
dataset_version
seed
model
parameter_count
loss
loss_weights
optimizer
learning_rate
batch_size
augmentation
training_time
PSNR
SSIM
LPIPS
OOD metrics
runtime
notes
```

## 18. Model selection

Select using a multi-axis table.

| Candidate | PSNR | SSIM | LPIPS | OOD | Runtime | Params | Failure profile |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline | | | | | | | |
| A | | | | | | | |
| B | | | | | | | |
| C | | | | | | | |

## 19. Do not claim an invented KLA score

Never state:

* official PSNR weight
* official SSIM weight
* official LPIPS weight
* official ID/OOD weight
* official latency target

unless KLA publishes it.

The current help document says those exact weightages are confidential.

## 20. Final evidence pack

Create:

* `metrics.csv`
* `ood_metrics.csv`
* `ablations.csv`
* `runtime.csv`
* `failure_cases.csv`
* visual panels
* experiment log
* final checkpoint manifest


## 21. Detailed experimental library retained from the previous knowledge base

The following sections preserve the earlier detailed experimental methods, naming conventions, ablation structure and evidence workflow. They are implementation guidance only and must be interpreted through the current source baseline above.

# KLA Winning Evaluation Strategy

## SEMICON India Hackathon 2026
## AI Based Restoration of Degraded Images for Semiconductor Inspection

## 0. Purpose

This document is the dedicated evaluation and winning strategy reference for the KLA problem.

It exists because the project needs a document that answers a question different from the implementation phases:

> How do we know that a solution is actually better, and what evidence would make the final submission convincing?

The existing Project.md defines the problem.
Architecture.md defines the system.
Rules.md defines engineering constraints.
Phases.md defines execution.
Design.md defines presentation.
memory.md preserves project state.
KLA_Problem_Deep_Dive_and_Drift_Sense_Transfer.md interprets the supplied organizer resources.

This document focuses specifically on evaluation quality, competitive positioning, evidence, model selection, and the path from an experimental result to a defensible claim.

---

# 1. Source status

The supplied KLA problem description explicitly names:

1. PSNR or pSNR.
2. SSIM.
3. LPIPS.
4. Before and after image comparisons.
5. Inference time.
6. Model size and training information.
7. In distribution testing.
8. Out of distribution testing.

It also says that the final evaluation script will be used as is by the KLA benchmarking team to measure quality and inference time on an H100 GPU.

The supplied material does NOT provide a complete official weighted scoring formula for these KLA restoration metrics.

Therefore:

* official facts must remain official facts
* internal ranking formulas are recommendations
* experimental composite scores must never be presented as KLA's official score

---

# 2. What winning should mean

Winning should not mean:

> the model has the highest PSNR on one validation split.

A stronger definition is:

> the final system produces highly faithful full resolution restorations across the documented degradation combinations and unseen source variation, while remaining fast, reproducible, explainable, and operationally reliable.

That definition reflects the actual KLA problem description.

The important dimensions are:

```text
reconstruction fidelity
        +
structural preservation
        +
OOD robustness
        +
combined degradation robustness
        +
inference efficiency
        +
reproducibility
        +
clear evidence
```

---

# 3. Evaluation hierarchy

Use the following hierarchy when judging a candidate.

## Tier 1: Correctness

Does the pipeline correctly map the degraded input to the correct target?

If no, stop.

## Tier 2: Baseline superiority

Does it beat simple interpolation?

If no, stop.

## Tier 3: Reconstruction quality

Does it improve PSNR, SSIM and LPIPS?

If no, investigate.

## Tier 4: Structural fidelity

Does it preserve thin structures, edges, gaps and repeated patterns?

If no, do not select it merely because global metrics improved.

## Tier 5: OOD robustness

Does the improvement survive a source or structure distribution change?

If no, the model may be memorizing.

## Tier 6: Combined degradation robustness

Does it remain stable when multiple degradation mechanisms appear together?

If no, it does not satisfy the central challenge.

## Tier 7: Runtime

Does it remain practical?

If no, compare the quality gain against the cost.

## Tier 8: Reproducibility

Can another machine run the exact evaluation path?

If no, the result is not submission ready.

---

# 4. The evaluation pyramid

Every serious candidate should be evaluated at five levels.

### Level A: Pixel

PSNR.

### Level B: Structure

SSIM.

### Level C: Perceptual difference

LPIPS.

### Level D: Local engineering evidence

Edge, thin feature, high frequency and error region analysis.

### Level E: Distribution robustness

OOD and combined degradation testing.

A model that wins only Level A is not automatically the best model.

---

# 5. Baselines are mandatory

The final model must be compared against simple alternatives.

At minimum:

```text
Nearest
Bilinear
Bicubic
Tiny neural baseline
Strong compact candidate
Final candidate
```

Bicubic is particularly important because it already solves the resolution increase component without learning.

The final model must demonstrate that learning provides meaningful improvement beyond interpolation.

---

# 6. Baseline table

Maintain a table like:

| Candidate | PSNR | SSIM | LPIPS | OOD PSNR | OOD SSIM | Runtime | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nearest | | | | | | | |
| Bilinear | | | | | | | |
| Bicubic | | | | | | | |
| Tiny CNN | | | | | | | |
| Candidate A | | | | | | | |
| Candidate B | | | | | | | |

Do not fill missing results with estimates.

---

# 7. Why PSNR alone is insufficient

PSNR is useful for pixel fidelity.

However, a model can obtain strong average pixel accuracy while:

* smoothing thin lines
* shifting edges
* removing small structures
* introducing ringing
* inventing high frequency details

Therefore PSNR should always be paired with structural and visual analysis.

---

# 8. Why SSIM matters

SSIM provides a structural perspective.

It is particularly useful when the important question is whether local contrast and structure are preserved.

However, it still does not prove physical correctness.

Use it as one component of the evidence package.

---

# 9. Why LPIPS matters but must be interpreted carefully

LPIPS is explicitly named in the KLA presentation requirements.

It can provide a complementary perceptual comparison.

But LPIPS should not automatically become a training loss.

A perceptual model trained on natural images can prefer plausible visual structure rather than exact semiconductor structure.

Therefore:

```text
LPIPS evaluation
!=
LPIPS training requirement
```

---

# 10. Metric sanity tests

Before trusting any leaderboard style result, test the metric implementation.

For identical images:

```text
prediction = ground truth
```

the metric behavior should be correct.

For intentionally degraded predictions:

```text
prediction = heavily blurred ground truth
prediction = noisy ground truth
prediction = shifted ground truth
```

the metrics should respond in an understandable direction.

This prevents an implementation mistake from becoming a false research conclusion.

---

# 11. Evaluation split design

At minimum maintain:

## Validation ID

Samples representative of the training distribution.

## Validation OOD

Samples separated by source or structural characteristics where possible.

## Stress set

Artificially difficult cases.

## Combined degradation set

Cases containing multiple degradation mechanisms.

These are internal evaluation sets unless the organizers define an official split.

---

# 12. Do not leak validation information

Never tune the final model directly against hidden test results.

If an official test set is released for final evaluation, treat it as a final benchmark.

Do not repeatedly modify the model based on test outputs.

Otherwise the test set becomes a training signal.

---

# 13. OOD evaluation

The KLA description explicitly says the test set includes samples from different sources than the training data.

Therefore OOD performance must be treated as a first class metric.

A useful internal table:

| Model | ID PSNR | OOD PSNR | ID SSIM | OOD SSIM | ID LPIPS | OOD LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Bicubic | | | | | | |
| Candidate A | | | | | | |
| Candidate B | | | | | | |

The most important observation is not only the absolute OOD score.

Also calculate:

```text
OOD drop = ID performance - OOD performance
```

A smaller degradation can indicate better generalization.

---

# 14. Combined degradation evaluation

Do not report only:

```text
speckle
Gaussian
super resolution
```

separately.

The official problem explicitly requires simultaneous handling.

Create:

```text
speckle only
Gaussian only
resolution only
speckle + resolution
Gaussian + resolution
speckle + Gaussian
speckle + Gaussian + resolution
```

where the dataset permits or synthetic controlled tests are appropriate.

---

# 15. Severity sweep

For controlled experiments, create degradation severity levels.

Example:

```text
weak
medium
strong
extreme
```

The exact numerical parameters must be calibrated from actual data or clearly labeled as stress-test settings.

Plot:

```text
quality
   |
   |\
   | \
   |  \
   |   \
   +---------- degradation severity
```

The purpose is to find the robustness curve rather than a single point.

---

# 16. Robustness curve

A model that performs extremely well at weak degradation but collapses immediately may be less useful than a model that is slightly worse at the easiest level but degrades gracefully.

Compare candidates across the severity range.

Record:

* PSNR
* SSIM
* LPIPS
* edge error
* failure count

---

# 17. Local error analysis

For every validation image, calculate at least:

```text
global PSNR
global SSIM
global LPIPS
```

Then create local diagnostics.

Useful regions include:

* high edge density
* thin structures
* strong contrast
* low contrast
* dense repeated patterns

The objective is to identify whether global quality hides local failure.

---

# 18. Thin structure evaluation

Create a subset with:

* narrow lines
* narrow gaps
* parallel structures
* small isolated structures
* high frequency patterns

Inspect:

```text
GT
input
prediction
absolute error
```

A candidate that improves global metrics but destroys thin structures should be rejected or investigated.

---

# 19. Edge displacement

An edge can be present but shifted.

That is different from simply being noisy.

Use edge maps to inspect:

```text
GT edge
prediction edge
```

Look for:

* missing edges
* added edges
* displaced edges
* broken edges
* thickened edges

---

# 20. Hallucination evaluation

The most dangerous restoration failure is creating structure that is not supported by the target.

Search for:

```text
input ambiguous
GT feature A
prediction feature B
```

A visually impressive prediction is not necessarily correct.

The final model should favor faithful reconstruction.

---

# 21. Smoothing failure

The opposite failure is excessive smoothing.

Symptoms:

* thin lines disappear
* gaps close
* corners become rounded
* contrast is reduced
* high frequency energy drops

Compare the model against bicubic and the target.

---

# 22. Ringing failure

Aggressive sharpening can create halos around edges.

Inspect strong transitions at high zoom.

If a model improves metrics but produces repeated halos, investigate the loss or architecture before selecting it.

---

# 23. Frequency diagnostics

Use FFT or another frequency diagnostic internally.

Compare:

```text
GT
input
prediction
```

Questions:

1. Is the prediction missing high frequency energy?
2. Is it creating excessive high frequency energy?
3. Are high frequency changes concentrated around real edges?
4. Does noise remain in the high frequency region?

This is a diagnostic, not an official scoring metric.

---

# 24. Error decomposition

For a candidate, classify failures into:

```text
noise residue
over smoothing
hallucination
ringing
edge shift
resolution recovery failure
intensity mismatch
OOD failure
```

Then count them.

A model with fewer severe structural failures can be preferable to one with slightly better aggregate metrics.

---

# 25. Ablation strategy

Ablations answer:

> Which part of our method actually causes the improvement?

At minimum investigate:

1. Architecture.
2. Loss.
3. Augmentation.
4. Synthetic degradation.
5. Output range policy.

Do not change five things simultaneously.

---

# 26. Loss ablation

Possible runs:

```text
L1
Charbonnier
L1 + SSIM
L1 + gradient
L1 + SSIM + gradient
```

The exact candidates are recommendations.

The final choice must come from measured evidence.

---

# 27. Augmentation ablation

Compare:

```text
official data only
+
mild calibrated augmentation
+
strong augmentation
```

The question is whether augmentation improves OOD performance without hurting ID fidelity.

---

# 28. Synthetic degradation ablation

Compare:

```text
no synthetic degradation
realistic calibrated synthetic degradation
overly broad synthetic degradation
```

This can reveal whether the synthetic generator helps generalization or simply introduces distribution mismatch.

---

# 29. Model size ablation

Compare at least:

```text
small
medium
large only if justified
```

Record:

* parameter count
* checkpoint size
* training time
* inference time
* PSNR
* SSIM
* LPIPS
* OOD metrics

---

# 30. Pareto analysis

The ideal candidate is not necessarily the highest quality.

Plot:

```text
runtime vs quality
parameters vs quality
```

A candidate on the Pareto frontier is one for which no alternative is simultaneously better in quality and cheaper in compute.

This is useful for selecting a practical final model.

---

# 31. Runtime benchmark protocol

The KLA material says inference time is benchmarked.

Therefore timing must be disciplined.

Use:

1. Fixed hardware.
2. Fixed software environment.
3. Fixed input set.
4. Warmup.
5. GPU synchronization where necessary.
6. Multiple repetitions.
7. Report median or another clearly defined statistic.
8. Separate model loading from inference when appropriate.

Do not compare timing measured under different conditions.

---

# 32. What runtime should include

There should be two measurements:

## Model-only

Useful for architecture research.

## End-to-end

Include:

```text
file loading
preprocessing
model inference
postprocessing
file saving
```

The submission path should be optimized based on the actual evaluator behavior.

---

# 33. H100 awareness

The supplied KLA description says the benchmark will use an H100 GPU.

This means CUDA compatibility and efficient inference matter.

Do not assume that the H100 means runtime is irrelevant.

A model can still waste time through:

* inefficient preprocessing
* unnecessary CPU transfers
* excessive model complexity
* repeated allocations
* poor batching
* unnecessary conversions

---

# 34. Model selection rule

Use a decision matrix:

| Criterion | Weight internally | Candidate A | Candidate B | Candidate C |
|---|---:|---:|---:|---:|
| ID fidelity | | | | |
| OOD robustness | | | | |
| Combined degradation | | | | |
| Structural fidelity | | | | |
| LPIPS | | | | |
| Runtime | | | | |
| Parameter count | | | | |
| Reproducibility | | | | |

The internal weights are not official KLA scoring weights.

They are only a decision aid.

---

# 35. Recommended internal score

If a single internal ranking number is useful, normalize each dimension first.

For example:

```text
internal_score =
  normalized_PSNR
+ normalized_SSIM
+ normalized_LPIPS_inverse
+ normalized_OOD
+ normalized_runtime_inverse
```

Do not present this as the KLA scoring formula.

Its only purpose is to prevent a decision from being made based on one metric.

---

# 36. Statistical stability

Do not trust a one run result if the difference is tiny.

If two candidates are nearly identical, repeat with controlled seeds.

Record:

```text
mean
standard deviation
best
worst
```

when practical.

The exact statistical protocol should reflect available compute.

---

# 37. Confidence in improvements

A claim such as:

> Model B is better.

should be supported by:

```text
multiple validation images
+
consistent metric improvement
+
visual evidence
+
OOD evidence
+
no unacceptable runtime regression
```

If the improvement appears only on one split, say so.

---

# 38. Checkpoint selection

Do not select a checkpoint based only on training loss.

Use a validation criterion that reflects the final objective.

A safer selection process is:

```text
save checkpoints
        |
        v
evaluate ID
        |
        v
evaluate OOD
        |
        v
inspect structural failures
        |
        v
select final candidate
```

---

# 39. Test set discipline

Once the official test set is released:

1. Run the frozen candidate.
2. Save outputs.
3. Record the exact commit.
4. Record the weight checksum.
5. Do not silently modify preprocessing.
6. Preserve the original outputs.

If a second candidate is tested, keep it clearly separated.

---

# 40. Evidence pack

The final evidence directory should contain:

```text
results/
    baseline_table.csv
    final_metrics.csv
    ood_metrics.csv
    degradation_sweep.csv
    runtime.csv
    ablations.csv
    failure_cases.csv
    figures/
        baseline_comparison.png
        best_cases.png
        difficult_cases.png
        ood_cases.png
        failure_cases.png
        runtime_quality.png
```

The exact filenames can differ.

The principle is that every presentation claim should map to an artifact.

---

# 41. Evidence provenance

For every result, know:

```text
dataset version
split
git commit
model checkpoint
configuration
seed
metric version
hardware
software environment
```

Without provenance, results become difficult to defend.

---

# 42. What makes a strong result slide

A strong result slide should answer:

1. Did it beat the baseline?
2. Does it generalize?
3. Does it preserve detail?
4. Is it fast?
5. Can we reproduce it?

Do not show ten unrelated charts.

Use a small number of decisive pieces of evidence.

---

# 43. Strong claim structure

Use:

```text
Claim
Evidence
Interpretation
Limitation
```

Example:

```text
Claim:
The final candidate improves restoration quality over bicubic.

Evidence:
PSNR, SSIM and LPIPS on the held out validation split.

Interpretation:
The learned model recovers information beyond interpolation.

Limitation:
The improvement is smaller on the OOD split.
```

This is more credible than a marketing claim.

---

# 44. Failure cases are evidence

A failure case should not be hidden.

Show:

```text
input
prediction
ground truth
```

and explain:

* what failed
* why it likely failed
* what was tested
* whether the failure was reduced

A technically honest failure analysis demonstrates understanding.

---

# 45. The strongest final model is not always the strongest final story

Suppose:

```text
Model A:
+0.2 dB PSNR
but
2x runtime
and
more hallucination
```

versus:

```text
Model B:
slightly lower PSNR
but
better OOD
faster
cleaner edges
```

Model B may be the stronger engineering choice.

The final decision must be evidence based.

---

# 46. Winning evidence checklist

Before finalizing the model:

[ ] Beats bicubic.

[ ] Handles documented resolution relationships.

[ ] Handles speckle.

[ ] Handles Gaussian noise.

[ ] Handles combined degradation.

[ ] Strong ID metrics.

[ ] Strong OOD metrics.

[ ] No obvious systematic hallucination.

[ ] No unacceptable over smoothing.

[ ] No severe ringing.

[ ] Runtime measured.

[ ] Model size recorded.

[ ] Training time recorded.

[ ] Evaluation script works cleanly.

[ ] Results reproducible.

[ ] Failure cases documented.

[ ] Presentation evidence generated.

---

# 47. Final competitive philosophy

The winning strategy is not:

```text
largest model
```

and not:

```text
highest single metric
```

It is:

```text
strong baseline
      +
correct data understanding
      +
controlled experiments
      +
high fidelity restoration
      +
OOD robustness
      +
combined degradation robustness
      +
efficient inference
      +
reproducibility
      +
clear evidence
```

That is the strongest defensible interpretation of the supplied KLA challenge.

---

# 48. Source boundary

The official KLA source is the supplied `description(2).txt`.

The separate Drift Sense material is not used as the KLA scoring formula.

Any internal score or weighting in this document is an engineering decision and must never be described as an organizer scoring rule.


# KLA Experiment, Ablation and Model Selection Playbook

## SEMICON India Hackathon 2026
## AI Based Restoration of Degraded Images for Semiconductor Inspection

## 0. Purpose

This document defines how to run experiments without losing scientific discipline.

The biggest danger in a hackathon is changing:

```text
model
loss
augmentation
preprocessing
optimizer
dataset
```

all at once and then claiming that the final model is better.

That creates an impressive looking result with no clear causal explanation.

This playbook enforces controlled experimentation.

---

# 1. Experimental principle

Every experiment should answer one question.

Bad:

> Train a new model with some better loss and augmentation and see what happens.

Good:

> Does adding gradient preservation to the baseline loss improve structural fidelity without harming OOD PSNR or runtime?

---

# 2. Experiment record

Every run should record:

```text
run_id
date
git_commit
dataset_version
split
random_seed
model
parameter_count
loss
loss_weights
optimizer
learning_rate
scheduler
batch_size
epochs
augmentation
synthetic_degradation
normalization
output_policy
training_time
validation_PSNR
validation_SSIM
validation_LPIPS
OOD_PSNR
OOD_SSIM
OOD_LPIPS
inference_time
checkpoint_path
notes
```

---

# 3. Experiment naming

Use:

```text
E001_bicubic
E002_tiny_cnn
E003_l1
E004_charbonnier
E005_l1_ssim
E006_l1_gradient
E007_combined_loss
E008_augmented
E009_ood
E010_model_size_small
E011_model_size_medium
E012_final_candidate
```

Do not overwrite old experiment results.

---

# 4. Experiment ladder

Use the following order:

```text
data correctness
      ↓
baseline
      ↓
tiny model
      ↓
loss
      ↓
architecture
      ↓
augmentation
      ↓
OOD
      ↓
runtime
      ↓
final candidate
```

This ordering minimizes confusion.

---

# 5. Experiment 0: pipeline sanity

Before training:

1. Load one pair.
2. Print shapes.
3. Print dtype.
4. Print range.
5. Visualize.
6. Run preprocessing.
7. Run model forward.
8. Calculate loss.
9. Save output.

No large training run is allowed before this succeeds.

---

# 6. Experiment 1: one sample overfit

Train on one pair.

Goal:

> prove that the model and training pipeline can learn the mapping.

If the model cannot overfit one pair, investigate:

* pairing
* preprocessing
* target shape
* loss
* output range
* learning rate
* architecture

Do not increase model size first.

---

# 7. Experiment 2: interpolation baselines

Measure:

```text
nearest
bilinear
bicubic
```

This creates the non learned reference.

---

# 8. Experiment 3: tiny neural baseline

Use a small residual CNN.

Goal:

* verify learnability
* verify training
* establish a neural reference

Do not optimize for the final score yet.

---

# 9. Experiment 4: loss ablation

Start from one fixed architecture.

Change only the loss.

Suggested runs:

```text
A: L1
B: Charbonnier
C: L1 + SSIM
D: L1 + gradient
E: L1 + SSIM + gradient
```

Keep:

* data
* optimizer
* learning rate
* schedule
* epochs
* seed

as constant as practical.

---

# 10. Experiment 5: architecture ablation

Once the loss is stable, compare architectures.

Candidate families can include:

* compact residual CNN
* residual super resolution architecture
* lightweight encoder decoder
* lightweight restoration blocks
* compact attention based architecture

These are research candidates, not organizer requirements.

---

# 11. Architecture selection criteria

Compare:

```text
quality
+
OOD robustness
+
parameter count
+
runtime
+
implementation reliability
```

Do not choose architecture based only on popularity.

---

# 12. Model size ladder

Compare:

```text
small
medium
large
```

The large model should only be retained if its improvement justifies its cost.

A model that adds complexity without meaningful improvement should be rejected.

---

# 13. Experiment 6: preprocessing ablation

Compare:

```text
raw floating point
global normalization
robust normalization
```

Measure:

* ID metrics
* OOD metrics
* structural artifacts

This is especially important because degraded values may exceed the target range.

---

# 14. Experiment 7: output range policy

Test:

```text
Policy A:
unbounded internal output + final conversion

Policy B:
bounded final output

Policy C:
explicit target range transformation
```

Do not clip the input blindly.

---

# 15. Experiment 8: augmentation

Compare:

```text
official data only
mild augmentation
calibrated augmentation
strong augmentation
```

The objective is not to maximize augmentation.

The objective is to improve generalization.

---

# 16. Experiment 9: synthetic degradation

Compare:

```text
no synthetic degradation
calibrated synthetic degradation
broad synthetic degradation
```

Measure whether synthetic augmentation:

* improves OOD
* improves combined degradation
* harms ID quality

---

# 17. Experiment 10: combined degradation

Evaluate the candidate on:

```text
speckle
Gaussian
resolution
speckle + resolution
Gaussian + resolution
speckle + Gaussian
all three
```

This is critical because the official challenge requires simultaneous handling.

---

# 18. Experiment 11: severity sweep

For each degradation create:

```text
weak
medium
strong
extreme
```

Record quality at every level.

This creates a robustness curve.

---

# 19. Experiment 12: OOD

Construct the strongest defensible OOD validation split.

Prefer source separation where metadata permits.

Do not call a random split OOD.

---

# 20. Experiment 13: failure analysis

For every candidate, identify:

* worst PSNR
* worst SSIM
* worst LPIPS
* largest edge error
* obvious hallucination
* strongest smoothing
* strongest ringing
* worst OOD case

Save visual panels.

---

# 21. Experiment 14: runtime

Measure:

```text
model load time
preprocessing
forward pass
postprocessing
file saving
```

Keep model only timing separate from end to end timing.

---

# 22. Experiment 15: precision

If mixed precision is considered, compare:

```text
FP32
mixed precision
```

Measure both quality and runtime.

Do not assume quality is unchanged.

---

# 23. Experiment 16: inference implementation

The final inference script must:

1. accept input directory
2. accept output directory
3. load weights
4. process all inputs
5. save restored outputs
6. require no manual edits

This matches the supplied KLA submission description.

---

# 24. Experiment 17: clean environment

Create a new environment.

Then:

```text
clone repository
install requirements
load model
run inference
verify outputs
```

Record the result.

---

# 25. Experiment 18: reproducibility

Repeat the final candidate under the documented seed and environment.

Compare results.

If the result changes significantly, investigate nondeterminism.

---

# 26. Ablation table

Maintain:

| Run | Change | PSNR | SSIM | LPIPS | OOD | Runtime | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| E003 | L1 | | | | | | |
| E004 | Charbonnier | | | | | | |
| E005 | + SSIM | | | | | | |
| E006 | + gradient | | | | | | |
| E007 | combined | | | | | | |

---

# 27. Architecture table

| Model | Params | PSNR | SSIM | LPIPS | OOD | Runtime | Artifacts |
|---|---:|---:|---:|---:|---:|---:|---|
| Tiny CNN | | | | | | | |
| Residual SR | | | | | | | |
| Lightweight candidate | | | | | | | |
| Candidate final | | | | | | | |

---

# 28. Pareto frontier

Plot:

```text
quality vs runtime
```

and:

```text
quality vs parameter count
```

Prefer candidates that are on the efficient frontier.

---

# 29. Decision rule

A candidate should become the final model only if:

```text
quality improvement
+
OOD improvement or stability
+
structural fidelity
+
acceptable runtime
+
reproducibility
```

are all acceptable.

---

# 30. Do not chase tiny metric gains

If Candidate A improves PSNR by an insignificant amount but:

* doubles runtime
* adds ringing
* hurts OOD
* complicates deployment

then the gain is not necessarily useful.

Document the tradeoff.

---

# 31. Do not chase visual sharpness

A sharper output is not automatically a better output.

Check:

```text
ground truth
prediction
absolute error
edge map
```

A model that creates false edges should be penalized internally.

---

# 32. Do not optimize against one image

A single spectacular example is not evidence.

Every meaningful claim should use a set.

Use:

* random samples
* difficult samples
* OOD samples
* combined degradation samples

---

# 33. Do not change validation after seeing results

Freeze the validation split before final model comparison.

If the split changes, treat it as a new experiment.

---

# 34. Do not silently change preprocessing

A common hidden source of improvement is changing normalization while thinking only the model changed.

Record preprocessing in every run.

---

# 35. Do not silently change the dataset

Record:

```text
dataset version
file count
split manifest
augmentation
synthetic generator version
```

---

# 36. Research questions worth answering

The strongest project will have answers to questions such as:

1. How much does learned restoration beat bicubic?
2. Which loss preserves structure best?
3. Does gradient loss help or merely sharpen?
4. Does synthetic degradation improve OOD?
5. How much does OOD performance drop?
6. Which degradation combination is hardest?
7. Which model size gives the best quality/runtime tradeoff?
8. What failure mode dominates?
9. Does mixed precision preserve quality?
10. Does the final inference script reproduce the research result?

---

# 37. Final model selection worksheet

Fill this only after experiments:

```text
Final model:
Architecture:

Parameters:

Checkpoint:

Training time:

Inference time:

PSNR:

SSIM:

LPIPS:

OOD PSNR:

OOD SSIM:

OOD LPIPS:

Best structural property:

Main failure mode:

Why this model was selected:

Why larger alternatives were rejected:

Why simpler alternatives were rejected:
```

---

# 38. Evidence mapping

Every final presentation claim should map to:

```text
claim
  ↓
experiment
  ↓
result file
  ↓
figure/table
  ↓
presentation slide
```

If there is no experiment behind a claim, remove or soften the claim.

---

# 39. Experiment freeze

Before final submission:

Freeze:

```text
architecture
loss
weights
preprocessing
augmentation
inference script
requirements
```

Record:

```text
git commit
checkpoint hash
environment
```

Then generate the final outputs.

---

# 40. Final experiment gate

Do not submit until:

[ ] baseline comparison complete

[ ] loss ablation complete

[ ] architecture comparison complete

[ ] augmentation comparison complete

[ ] combined degradation evaluation complete

[ ] OOD evaluation complete

[ ] failure analysis complete

[ ] runtime benchmark complete

[ ] final checkpoint frozen

[ ] clean environment test passed

[ ] evidence pack generated

---

# 41. Source boundary

The supplied KLA description defines the official restoration task and the named evaluation information.

The Drift Sense webinar is a separate problem statement.

It can inform domain aware experimentation, but its coordinate scoring, 30 sample requirement, image dimensions and registration task must not be copied into KLA experiments.


# KLA Data Forensics and Degradation Engineering

## SEMICON India Hackathon 2026
## AI Based Restoration of Degraded Images for Semiconductor Inspection

## 0. Purpose

This document is the dedicated data engineering reference.

The central principle is:

> Do not design the restoration model around assumptions about the data when the actual paired dataset can be inspected.

The supplied KLA description gives a high level contract:

* grayscale images
* paired degraded and clean images
* documented full resolution examples of 512x512 and 256x256
* corresponding degraded examples of 256x256 and 128x128
* speckle noise
* Gaussian noise
* spatial resolution reduction
* combined degradation
* intensity values that may exceed the ground truth range
* diverse data origins
* in distribution and out of distribution evaluation

This document converts those facts into a rigorous data audit and degradation engineering workflow.

---

# 1. Source derived facts

The supplied KLA description states that:

1. Ground truth is clean and full resolution.
2. Degraded input is noisy and downsampled.
3. Grayscale single channel images are used.
4. Degraded intensity values may exceed the ground truth range.
5. Multiple degradation types can occur together.
6. Different source structures are present.
7. The test set includes OOD data.

These are source derived facts.

Everything beyond those facts in this document is an engineering recommendation unless explicitly labeled otherwise.

---

# 2. First rule: inspect before preprocessing

Do not begin with:

```python
image = image / 255.0
```

Do not begin with:

```python
image = np.clip(image, 0, 255)
```

Do not begin with:

```python
image = cv2.normalize(...)
```

Do not begin with:

```python
image = image.astype(np.uint8)
```

First determine what the files actually contain.

---

# 3. Dataset inventory

Create an inventory with:

```text
path
filename
extension
file size
image width
image height
channels
dtype
minimum
maximum
mean
standard deviation
```

For paired data also record:

```text
input path
target path
input dimensions
target dimensions
pair identifier
```

---

# 4. Shape audit

Verify every documented resolution relationship.

Expected examples from the problem description:

```text
256x256 -> 512x512
128x128 -> 256x256
```

Do not assume these are the only shapes if the actual dataset contains additional cases.

If unexpected shapes occur:

1. report them
2. inspect them
3. determine whether they are legitimate
4. do not silently resize them

---

# 5. Channel audit

The source description states that images are grayscale and single channel.

Verify this in the actual files.

Possible representations include:

```text
H x W
H x W x 1
```

If the files contain unexpected channels, stop and inspect them.

Do not silently discard channels.

---

# 6. Dtype audit

Record:

```text
uint8
uint16
float32
float64
```

or whatever is actually present.

Dtype can change the interpretation of the intensity range.

A float image with values around 0 to 1 is not equivalent to a uint16 image with values around 0 to 65535.

---

# 7. Intensity range audit

For every input and target distribution record:

```text
min
max
mean
median
std
p01
p05
p25
p50
p75
p95
p99
```

Also record:

```text
fraction below expected lower range
fraction above expected upper range
```

This is particularly important because the KLA description explicitly says degraded values can exceed the ground truth range.

---

# 8. Never clip before measuring

Clipping destroys evidence.

The audit should operate on the raw loaded values.

Only later should a preprocessing experiment decide whether clipping is useful.

---

# 9. Global versus per image normalization

Compare:

## Global normalization

One transform for the whole dataset.

Advantages:

* consistent scale
* preserves relative differences across images

Risks:

* extreme values can dominate

## Per image normalization

Normalize every image independently.

Advantages:

* stabilizes image ranges

Risks:

* destroys absolute intensity information
* may make different structures artificially similar

Do not choose per image normalization automatically.

---

# 10. Range policy experiments

Evaluate at least:

```text
Policy A:
raw floating point values

Policy B:
global dataset normalization

Policy C:
global robust normalization

Policy D:
normalization plus explicit output conversion
```

The exact formulas are experimental.

The final policy should be selected using both metrics and structural inspection.

---

# 11. Pair integrity

For every pair verify:

1. Correct filename association.
2. Correct source identifier.
3. Correct resolution relationship.
4. Correct spatial correspondence.
5. No accidental shuffling between inputs and targets.

A restoration model can appear to train normally even with a pairing bug.

---

# 12. Pair visualization

Randomly select paired samples and display:

```text
degraded
ground truth
upsampled degraded
difference
```

Check:

* same scene
* same structure
* same orientation
* expected resolution relationship

Do this before model development.

---

# 13. Duplicate detection

Check for:

* exact duplicate targets
* exact duplicate inputs
* near duplicate pairs

Duplicates can cause leakage between training and validation.

If duplicate detection is computationally expensive, start with hashes and then use image similarity for suspicious groups.

---

# 14. Split integrity

A random split may be insufficient if multiple images originate from the same source.

If source identifiers exist, prefer grouping by source.

The goal is to prevent nearly identical structures from appearing in both training and validation.

---

# 15. OOD split philosophy

The source description says OOD samples come from different sources.

Therefore the strongest internal OOD split should ideally separate source identity.

If source metadata is unavailable, use defensible structural grouping and document the limitation.

Do not call a random validation split OOD.

---

# 16. Spatial statistics

Characterize the images using:

* edge density
* connected component statistics where meaningful
* local contrast
* intensity entropy
* gradient magnitude
* frequency energy

These are not official KLA metrics.

They help understand the data distribution.

---

# 17. Edge density

Calculate a simple edge map and estimate:

```text
edge_pixels / total_pixels
```

Compare across sources.

This can identify whether the training set is dominated by smooth regions or dense structures.

---

# 18. Local contrast

Measure local contrast across patches.

This helps identify:

* low contrast regions
* high contrast edges
* dense structures

A restoration model should not be evaluated only on the average image.

---

# 19. Frequency distribution

Compute a frequency representation for representative images.

Compare:

```text
ground truth
degraded input
```

The goal is to understand what spatial frequencies are lost or corrupted.

---

# 20. Residual analysis

If the degraded image can be resized into target space, calculate an approximate residual:

```text
target - upsampled_degraded
```

This is not the exact physical degradation.

It is a diagnostic.

Analyze:

* residual mean
* residual variance
* residual histogram
* residual spatial distribution
* residual frequency distribution

---

# 21. Why residual analysis matters

It helps answer:

> Is the problem mostly noise removal, mostly missing high frequency detail, or a combination?

The answer can influence architecture and loss design.

---

# 22. Speckle analysis

The supplied description characterizes speckle as random pixel level noise and explicitly notes that it can push intensity beyond the ground truth range.

Investigate whether the observed corruption appears:

* additive
* multiplicative
* intensity dependent
* spatially varying

Do not assume the answer.

Measure it.

---

# 23. Gaussian noise analysis

Investigate whether the observed Gaussian like corruption produces:

* uniform variance
* intensity dependent variance
* edge dependent behavior

Again, the source description gives the degradation category, not a complete parameterization.

The actual dataset should determine how strongly a synthetic approximation is needed.

---

# 24. Resolution loss analysis

The documented examples show a two times spatial resolution reduction.

Investigate:

* aliasing
* blur
* line merging
* gap closing
* thin feature disappearance

Downsampling is not simply a change in array dimensions.

The interpolation or sampling process affects what information survives.

---

# 25. Information loss versus noise

Separate:

```text
information still present but corrupted
```

from:

```text
information no longer present after downsampling
```

This distinction is central to restoration.

Noise can often be reduced.

Completely lost spatial information must be inferred.

---

# 26. Synthetic degradation should be calibrated

The separate Drift Sense webinar is useful as a methodological reference because it emphasizes controlled programmatic synthetic generation.

For KLA, the official paired dataset is primary.

Synthetic degradation should therefore be used to:

1. stress test
2. augment
3. study controlled behavior
4. create difficult cases

It should not replace the official dataset by default.

---

# 27. Synthetic data pipeline

A controlled synthetic pipeline should look like:

```text
clean image
     |
     v
controlled resolution reduction
     |
     v
controlled noise
     |
     v
optional additional calibrated corruption
     |
     v
synthetic degraded image
```

Keep the clean source unchanged.

Store the parameters used.

---

# 28. Degradation order experiment

Test:

```text
clean -> downsample -> noise
```

against:

```text
clean -> noise -> downsample
```

The actual official process is not established by the supplied description.

This is therefore an experiment.

---

# 29. Noise strength sweep

For each synthetic degradation define:

```text
weak
medium
strong
extreme
```

Store the exact parameter.

Do not use an undocumented random strength.

---

# 30. Combined degradation matrix

Use a controlled matrix:

| Case | Speckle | Gaussian | Downsample |
|---|---:|---:|---:|
| 1 | no | no | yes |
| 2 | yes | no | yes |
| 3 | no | yes | yes |
| 4 | yes | yes | yes |
| 5 | weak | weak | yes |
| 6 | medium | medium | yes |
| 7 | strong | weak | yes |
| 8 | weak | strong | yes |
| 9 | strong | strong | yes |

These are stress-test categories, not official scoring categories.

---

# 31. Calibration against real data

Compare synthetic degraded images against actual degraded images using:

* intensity histograms
* residual histograms
* gradient statistics
* frequency statistics
* visual panels

If synthetic data looks much noisier or cleaner than the real data, reduce or redesign the augmentation.

---

# 32. The danger of unrealistic augmentation

A model can become excellent at removing synthetic noise that never appears in the real challenge.

That produces:

```text
synthetic benchmark improvement
but
real benchmark degradation
```

Therefore synthetic augmentation must earn its place through validation.

---

# 33. Semiconductor structure awareness

The Drift Sense webinar describes structured semiconductor layouts rather than natural photographs.

It discusses repeated patterns, Manhattan geometry, memory and logic structures.

For our restoration problem, this suggests useful stress categories:

* repeated lines
* periodic structures
* dense arrays
* sharp corners
* thin lines
* narrow gaps
* high contrast transitions

This is domain informed testing, not an official KLA synthetic data requirement.

---

# 34. Repeated structure stress test

Create cases where:

```text
pattern A
pattern A
pattern A
pattern B
```

The local difference in B is important.

The model should not simply reconstruct the dominant repeated pattern.

---

# 35. Thin line stress test

Use cases where a thin line becomes partially or fully ambiguous after downsampling.

Measure whether the model:

* removes it
* recreates it correctly
* invents it incorrectly
* thickens it

---

# 36. Narrow gap stress test

A narrow dark gap between two bright structures can disappear after downsampling.

Test whether restoration:

* preserves the gap
* closes it
* creates multiple gaps
* produces ringing

---

# 37. Corner stress test

Inspect:

* right angle corners
* intersections
* T junctions
* closely spaced corners

Look for rounding or artificial sharpening.

---

# 38. Intensity excursion stress test

Because input values may exceed target range, construct cases with extreme input values and verify that preprocessing does not destroy them.

Record:

```text
raw min/max
processed min/max
model input min/max
prediction min/max
```

---

# 39. Data quality gate

Do not begin large scale training until:

[ ] pairing verified

[ ] shape verified

[ ] channel verified

[ ] dtype verified

[ ] range verified

[ ] duplicates investigated

[ ] split defined

[ ] ID validation defined

[ ] OOD validation defined

[ ] baseline data loaders tested

---

# 40. Data artifacts

Save:

```text
data_audit.csv
pair_integrity.csv
distribution_summary.csv
source_summary.csv
split_manifest.csv
synthetic_degradation_config.json
synthetic_examples/
```

The exact names can differ.

The important point is reproducibility.

---

# 41. Data lineage

Every training result should be traceable to:

```text
dataset version
split manifest
preprocessing configuration
augmentation configuration
```

A model checkpoint without data lineage is incomplete.

---

# 42. Final data philosophy

The strongest data strategy is:

```text
inspect real data
      |
      v
measure distributions
      |
      v
establish clean validation
      |
      v
calibrate synthetic degradation
      |
      v
stress test
      |
      v
train
      |
      v
re-evaluate on real validation
```

Do not reverse this order.

---

# 43. Source boundary

The KLA description is authoritative for the actual restoration task.

The Drift Sense material is useful only for transferable imaging methodology.

Its specific synthetic image dimensions, coordinate task, scoring rubric and coordinate outputs must not be imported into KLA.
