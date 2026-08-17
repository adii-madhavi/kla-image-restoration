# Submission_and_Repository.md
# KLA Phase 1 Submission, GitHub, Inference and Reproducibility


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



## 1. Mandatory Phase 1 package

The latest help document specifies:

1. Solution presentation
2. GitHub repository
3. Standalone inference script
4. Reproducible training code
5. Model weights/config
6. README
7. Dependencies/environment
8. Results/output samples

## 2. Inference script

The script must:

1. accept input directory
2. accept output directory
3. discover degraded input images
4. load weights/config
5. restore every image
6. save each output
7. run on NVIDIA GPU
8. support batching where possible
9. require no manual edits

Recommended invocation:

```bash
python inference.py --input_dir /path/to/input --output_dir /path/to/output
```

The exact names and image format must follow official evaluator instructions.

## 3. Output scoring

KLA scores the raw saved output.

There is no server-side clipping or normalization.

Therefore final range formatting is our responsibility.

Do not leave output range unspecified.

## 4. Training code

The submitted training code must be sufficient to reproduce the submitted checkpoint.

Include:

* model definition
* dataset definition
* preprocessing
* augmentation
* losses
* optimizer
* scheduler
* training loop
* checkpoint selection
* seed/configuration

## 5. Environment

Provide a usable environment specification.

Accepted examples in the supplied guidance include:

* `requirements.txt`
* `environment.yml`
* Docker configuration

The environment should cover training and inference dependencies as appropriate.

## 6. README

The README should make the repository self-service.

Minimum sections:

1. Problem
2. Repository structure
3. Environment
4. Weight acquisition
5. Dataset expectations
6. Inference command
7. Training command
8. Output contract
9. Results
10. Runtime
11. Hardware
12. External resource disclosure
13. Limitations

## 7. Recommended repository structure

```text
repository/
    README.md
    requirements.txt
    train.py
    inference.py
    configs/
    src/
    weights/
    results/
    references/
    presentation/
```

## 8. Weight packaging

Supported concepts from the source materials include:

* `.pt`
* `.pth`
* `.onnx`
* other usable model formats

The exact file must match the implemented inference loader.

If too large for normal Git storage, provide an accessible download method allowed by the portal and document the checksum.

## 9. Results/output samples

Keep:

* metric summary
* representative restored images
* failure cases
* OOD examples where possible

## 10. External resource disclosure

For each external model/dataset:

* name
* source URL
* license
* paper/model card/dataset card
* usage

## 11. Clean-machine test

From a fresh environment:

```text
clone
install
obtain weights
prepare input directory
run inference
inspect output
record runtime
```

No local source edits.

## 12. Runtime recording

Record the full pipeline.

Do not report only neural-network forward pass.

## 13. Final submission naming

The older website wording and current template have slightly different format language.

The current template explicitly supplies a filename pattern:

`Team Name_PSNo`

and example:

`i4C_PS01`

The participant notice also says to follow the portal naming convention.

Therefore final naming should be checked against the portal at upload time rather than hard-coded in the repository.

## 14. Submission freeze

Before uploading:

[ ] public GitHub accessible

[ ] inference works

[ ] training code reproducible

[ ] weights accessible

[ ] README complete

[ ] dependencies complete

[ ] output examples included

[ ] PDF opens

[ ] six/seven slide rule obeyed

[ ] links work

[ ] all external resources disclosed

[ ] clean environment test passed

## 15. Later evaluation

The supplied materials say shortlisted submissions may be run on KLA hidden test data and benchmarked on a common H100.

Do not retrain on hidden test inputs unless KLA later explicitly permits it.
