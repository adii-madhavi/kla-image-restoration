# Design.md

# SEMICON India Hackathon 2026 --- KLA

# Phase 1 PPT/PDF Design, Visual System and Evidence Assembly

## 0. Design objective

The deck is not a generic research presentation. It is a compact
technical case for selection. It must prove that the team understands
the degradation process, has built a reproducible restoration pipeline,
has measured it correctly, has considered hidden-test generalization and
H100 inference throughput, and has a repository evaluators can actually
run.

The supplied template's core idea-detail pointers must be preserved. The
template instruction says 6--7 slides maximum including the title slide,
and its instruction slide must be removed. Prefer points, diagrams,
infographics, tables and image comparisons over paragraphs.

## 1. Visual design system

### General style

Use the organizer template's visual language as the base. Recommended
treatment is technical, clean and semiconductor-oriented rather than
decorative.

### Color system

Use a restrained palette:

-   deep navy for primary structure
-   technical blue for process/model elements
-   restrained orange for degradation/warnings/highlights
-   white/light grey/dark grey neutrals

Do not introduce a different accent color for every slide.

### Typography

Use one clean sans-serif family. Maintain a hierarchy of title, section
heading, body and caption. Do not solve density problems by shrinking
text until it becomes unreadable.

### Visual hierarchy

Each slide should have:

1.  one dominant message
2.  one dominant visual
3.  three to five supporting elements where appropriate
4.  consistent alignment and spacing

## 2. Recommended final deck: 7 slides

### Slide 1 --- Title + Team + One-Line Solution

**Objective:** within 10 seconds the evaluator should understand what
problem was solved, what the model does, why the approach is credible
and who built it.

Content:

-   team name
-   team leader
-   members
-   college
-   contact
-   official problem number if provided
-   title: **AI-Based Restoration of Degraded Images for Semiconductor
    Inspection**
-   one-line technical concept
-   simple flow:
    `NoisyLR input → learned restoration model → restored high-resolution output`
-   visible degradation labels: speckle, Gaussian, downsampling, unknown
    degradation order

Do not claim state-of-the-art without evidence.

### Slide 2 --- Problem Understanding + Dataset + Why It Matters

Show:

-   semiconductor inspection context
-   formal input/output task
-   paired GT and degraded data
-   degradation types
-   documented dimensions
-   input/output value behavior
-   ID versus dissimilar/OOD test framing

Use a visual transformation rather than paragraphs:

``` text
Degraded image
  + noise + resolution loss
          ↓
     restoration
          ↓
Clean ground truth
```

Highlight:

-   NoisyLR can contain values outside `[0,1]`
-   GT is in `[0,1]`
-   images are normalized a priori
-   degradation may occur in any order

### Slide 3 --- Proposed Solution: End-to-End Pipeline + Architecture

Use a large left-to-right pipeline:

``` text
Paired data
 ↓
Audit / split
 ↓
Preprocess
 ↓
Calibrated augmentation
 ↓
Restoration model
 ↓
2× reconstruction
 ↓
Output
```

Below it show the candidate architecture and why it fits. Include
baselines in the progression rather than presenting the final model as
if it appeared from nowhere.

Recommended candidate progression:

`Bicubic → tiny neural baseline → stronger candidate models → Pareto-selected final model`

Do not put code on the slide.

### Slide 4 --- Training Methodology + Losses + Augmentation + Reproducibility

Show four connected blocks:

``` text
Data
 ↓
Training
 ↓
Loss
 ↓
Validation / checkpoint
```

Include:

-   ID/OOD split strategy
-   patch/crop strategy if used
-   L1/Charbonnier baseline
-   structural/gradient ablations
-   safe paired geometric augmentation
-   calibrated synthetic degradation
-   seed/checkpoint/logging hygiene

### Slide 5 --- Experiments + Metrics + Ablation + Runtime

Main quantitative table should compare baseline and final/candidates.
Suggested columns:

  -------------------------------------------------------------------------------
  Method   Training   Loss       Params   PSNR/pSNR      SSIM     LPIPS   Runtime
           data                                                         
  -------- ---------- ------- --------- ----------- --------- --------- ---------

  -------------------------------------------------------------------------------

Also show:

-   loss ablation result
-   augmentation ablation result
-   model-size/throughput result
-   ID/OOD comparison

Runtime must state the timing scope. Ideally include:

-   disk read
-   preprocessing
-   host-to-device
-   model execution
-   device-to-host
-   postprocessing
-   disk save

### Slide 6 --- Visual Results + Failure Analysis + Innovation + Impact

Use three evidence cases:

1.  typical success
2.  hard/noisy case
3.  failure/limitation

For each visual comparison use the same crop and order:

`Degraded | Restored | Ground Truth`

Optionally add a zoomed detail and absolute-error panel.

Innovation should be evidence-based. Examples of credible framing
include:

-   degradation-aware training
-   explicit OOD validation
-   structural-fidelity losses
-   joint restoration rather than isolated denoising/super-resolution
-   quality/throughput Pareto selection
-   reproducible inference

Impact must not claim production deployment unless it actually occurred.

Limitations should be explicit, including irrecoverable information
loss, dependence on degradation coverage, hallucination risk and unknown
hidden-test score.

### Slide 7 --- Feasibility + Repository + Reproduction + References + Next Steps

Show:

-   PyTorch / NumPy / metric stack
-   CUDA / NVIDIA GPU context
-   CLI-based inference
-   no manual source edits
-   GitHub QR code and clickable link if supported
-   repository tree
-   exact tested reproduction commands
-   external resource disclosure
-   3--4 high-value references
-   next steps

Example commands must be replaced by the actual tested commands before
submission:

``` bash
python train.py --config configs/final.yaml
python inference.py --input_dir <input> --output_dir <output> --config configs/final.yaml --weights weights/final.pt
```

## 3. Mapping to organizer pointers

  Organizer pointer                        Final slide
  ---------------------------------------- -------------
  Team Details                             1
  Problem Statement Addressed              2
  Idea Description                         2--3
  Proposed Solution                        3--4
  Innovation and Uniqueness                6
  Impact and Benefits                      6
  Technology & Feasibility / Methodology   3--4 and 7
  GitHub & Video Link                      7
  Research & References                    7

## 4. Visual rules

-   Use actual challenge data where permitted.
-   Use identical crops in comparisons.
-   Label every image.
-   Label units on charts.
-   Never fabricate a metric.
-   Avoid decorative AI imagery when a technical diagram communicates
    better.
-   Use simple tables, bars and line charts.
-   Avoid pie charts, 3D charts and decorative radar charts.
-   Keep whitespace.
-   Do not use paragraphs when a diagram can communicate the idea.

## 5. Evidence assets that must exist before final slide assembly

Prepare:

### Data

-   dataset summary
-   input/GT dimensions
-   range statistics
-   source/split summary

### Metrics

-   PSNR/pSNR
-   SSIM
-   LPIPS
-   ID/OOD comparison

### Runtime

-   hardware
-   batch size
-   warm-up policy
-   timing scope
-   ms/image

### Visuals

-   degraded/restored/GT panels
-   zoomed crops
-   error maps
-   at least one failure case

### Reproducibility

-   Git commit
-   checkpoint
-   requirements
-   exact inference command

## 6. Final presentation review rubric

### Technical understanding

Can the judge see that the team understands degradation and the inverse
problem?

### Experimental credibility

Are improvements backed by baselines and ablations?

### Generalization

Is OOD behavior explicitly shown?

### Compute

Is runtime measured correctly?

### Reproducibility

Can the repository actually be run?

### Visual communication

Can the deck be understood quickly without reading paragraphs?

### Honesty

Are limitations and failures visible?

## 7. Final slide-build sequence

1.  Freeze final model and metrics.
2.  Generate final evidence assets.
3.  Fill slide 2 with verified problem facts.
4.  Fill slide 3 with the actual architecture.
5.  Fill slide 4 with actual training/loss/augmentation decisions.
6.  Fill slide 5 with measured tables.
7.  Fill slide 6 with measured visual results and limitations.
8.  Fill slide 7 with tested repository commands and references.
9.  Add team details.
10. Remove template instruction slide.
11. Export PDF.
12. Open the PDF independently and inspect every page.
13. Verify links/QR code.
14. Verify no placeholder remains.
15. Verify every metric matches an artifact in `results/`.

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


## Current template reconciliation

The current template contains these pointer sections:

1. Team Details
2. Problem Statement Addressed
3. Idea Description
4. Proposed Solution
5. Innovation and Uniqueness
6. Impact and Benefits
7. Technology and Feasibility
8. GitHub and Video Link
9. Research and References

The template itself contains an instruction slide that must be removed and limits the final deck to six or seven slides including the title.

The KLA help document separately provides a detailed 12-slide content recommendation.

### Recommended six-slide strategy

If six slides are required:

1. Team + one-line solution
2. Problem + dataset + degradation
3. Idea + end-to-end pipeline
4. Architecture + preprocessing + loss + augmentation
5. Results + innovation + impact + failure
6. Technology + runtime + GitHub + research/references

### Recommended seven-slide strategy

If seven slides are allowed:

1. Team + one-line solution
2. Problem understanding
3. Dataset + degradation analysis
4. Proposed solution + architecture
5. Training + losses + augmentation
6. Results + runtime + failure + impact
7. Technology + repository + references

The exact final slide assignment should remain within the supplied template layouts.

## Mandatory visual rules

* Avoid paragraphs.
* Use concise text.
* Prefer diagrams and images.
* Show full-resolution examples where possible.
* Use matching crops and identical display conditions for comparison.
* Keep numerical metrics linked to a named split.
* Do not use fake measurements.
* Do not present hidden KLA results as if they were available.

## Required evidence before deck assembly

### Data

* GT/NoisyLR pair examples
* range/histogram plot
* degradation illustration

### Model

* architecture diagram
* training pipeline

### Results

* PSNR
* SSIM
* LPIPS
* baseline comparison
* OOD comparison
* runtime

### Failure

* at least one failed example
* cause
* attempted mitigation

### Reproducibility

* repository structure
* inference CLI
* environment
* weight location

## Template page roles

The supplied template visually uses a dark semiconductor background, sponsor logos and green accent elements.

Keep the organizer visual system.

Do not replace it with an unrelated generic slide design.

## Presentation narrative

The final deck should communicate:

`problem -> data -> solution -> evidence -> efficiency -> reproducibility`

not:

`buzzwords -> giant architecture -> screenshots`.
