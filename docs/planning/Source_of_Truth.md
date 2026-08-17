# Source_of_Truth.md
# Current KLA Source Hierarchy and Conflict Log

## 1. Current source bundle

Primary current files supplied for this rebuild:

1. `KLA_Problem Statement_Studen help document.pdf`
2. `Idea Submission Template_Hackathon 2026.pptx`
3. `KLA Problem Statement_explanation(1).pptx`
4. `q&A session transcript.txt`
5. `problemstatement.txt`
6. `combinedreport.txt`
7. Current participant notice supplied with the dataset link and Phase 1 deadline

The previous MD package is reference only.

## 2. Authority hierarchy

Use this order when sources conflict:

1. Current participant help document
2. Current organizer submission template for presentation format
3. Latest KLA Q&A
4. Original KLA presentation
5. Original KLA webinar transcript
6. Current participant notice
7. Generated summaries
8. Older generated MDs

## 3. Conflict: slide count

Current template: six or seven slides including the title.

Earlier website wording: eight or nine slides.

Working rule: six or seven slides.

Re-check the live portal before upload in case the website has been updated.

## 4. Conflict: PPT/PDF wording

The current help document's Phase 1 table calls the presentation a PPT/PPTX deliverable.

The current organizer template says to save the solution as PDF and that PPT, Word or other formats will not be supported.

Working rule:

* author in the supplied PPT template
* export the final portal artifact as PDF
* re-check the portal before upload

## 5. Conflict: twelve recommended content sections versus six/seven slides

The help document provides a 12-part recommended content structure.

The template caps the final deck at six or seven slides.

Working rule:

Compress the twelve content areas into the six or seven template slides without changing the underlying pointer categories.

## 6. Conflict: blur

The original KLA educational presentation and transcripts use blur as a general example of image degradation and show illustrative degradation chains.

The latest help document explicitly limits the benchmark to:

1. Gaussian noise
2. Speckle noise
3. Downsampling

Working rule:

Blur is contextual, not a fourth benchmark corruption.

## 7. Conflict: L1/L2 versus official metrics

Earlier generated summaries discuss L1/L2 as losses and may mention them alongside evaluation discussions.

Current official quality reporting is based on PSNR, SSIM and LPIPS with a confidential fixed combination.

Working rule:

L1/L2 are training candidates.

PSNR/SSIM/LPIPS are the official reported image-quality metrics.

## 8. Current hard facts

* GT `[0,1]`
* NoisyLR may extend outside `[0,1]`
* hidden test exposes degraded inputs only
* KLA retains hidden GT
* ID and OOD image content are evaluated
* OOD keeps the same three degradation mechanisms
* public external data and pretrained weights are allowed under suitable licenses
* H100 benchmark
* end-to-end runtime includes I/O and transfers
* batch processing preferred
* training hygiene is evaluated
* standalone inference required
* training reproducibility required
* Phase 1 deadline 16 August 2026

## 9. Future update protocol

When a new organizer update arrives:

1. save the original
2. read it completely
3. compare against this file
4. record conflicts
5. update `Project.md`
6. update `Architecture.md`
7. update `Phases.md`
8. update `Evaluation_and_Model_Selection.md`
9. update `Submission_and_Repository.md`
10. update `Design.md`
11. re-run the final package audit
