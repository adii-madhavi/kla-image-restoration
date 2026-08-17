# CLAUDE.md
# KLA Hackathon Master Operating Instructions

## 0. Read first

Before any substantive change, read:

1. `Source_of_Truth.md`
2. `Project.md`
3. `Architecture.md`
4. `Phases.md`
5. `Evaluation_and_Model_Selection.md`
6. `Submission_and_Repository.md`
7. `Design.md`
8. `Research_and_Resources.md`

## 1. Task boundary

This repository solves:

`degraded noisy low-resolution grayscale image -> restored full-resolution image`

Official benchmark degradations:

1. Additive Gaussian noise
2. Multiplicative speckle noise
3. Spatial downsampling

The order may vary.

Do not import the Applied Materials Drift Sense problem into this project.

## 2. Data

GT is normalized to `[0,1]`.

NoisyLR may extend slightly outside `[0,1]`.

Do not blind clip before deciding and measuring the effect.

Use official dataset dimensions and file conventions.

## 3. Model

Any suitable restoration architecture is allowed.

CNN, transformer, algorithm unrolling, published architectures and justified hybrids are valid.

Public pretrained weights and datasets are allowed when properly licensed and disclosed.

## 4. Evaluation

Official quality metrics:

* PSNR
* SSIM
* LPIPS

KLA uses a fixed internal combination with confidential weights.

Other evaluation axes:

* end-to-end throughput
* training and compute hygiene

Benchmark GPU:

NVIDIA H100.

## 5. Inference

The final inference script must accept input and output directories, process all degraded images, save the required outputs and work without source edits.

End-to-end time matters, including I/O and transfers.

Batch processing is strongly preferred.

## 6. Experiment discipline

Baseline first.

Overfit one or two pairs.

Change one major component at a time for ablation claims.

Track seeds, configuration, checkpoints and results.

Inspect images as well as metrics.

## 7. Presentation

Use the supplied organizer template.

Current template limit:

six or seven slides including the title.

Remove the instruction slide.

Final submission format should follow the latest template/portal instruction, with the current template saying PDF only.

## 8. Source fidelity

When current sources conflict, consult `Source_of_Truth.md` before changing implementation or presentation requirements.

Never fabricate an organizer requirement, metric weight, latency target or hidden-test detail.
