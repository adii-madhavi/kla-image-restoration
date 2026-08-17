# presentation/

Place the final Phase 1 slide deck source and exported PDF here
(`presentation/final.pptx`, `presentation/final.pdf`).

## Constraints (see Source_of_Truth.md for the full conflict log)

- **Slide count:** 6-7 slides including the title (current organizer
  template — stricter of two conflicting sources; re-check the live
  portal before upload).
- **Format:** author in the supplied organizer PPT template; the
  current template says the final portal upload must be **PDF**, not
  PPT/PPTX/Word. Re-check the portal at upload time — the help
  document's deliverables table separately says PPT/PPTX, which is a
  known wording conflict.
- **Content:** remove the instruction slide; prefer points, diagrams,
  infographics and image comparisons over paragraphs (Design.md §0).

## Recommended slide structure (Design.md §2)

1. Title + team + one-line solution
2. Problem understanding + dataset + why it matters
3. Proposed solution: end-to-end pipeline + architecture
4. Training methodology + losses + augmentation + reproducibility
5. Experiments + metrics + ablation + runtime
6. Visual results + failure analysis + innovation + impact
7. Feasibility + repository + reproduction + references + next steps

## Evidence assets to pull in before final assembly

- `results/metrics/` — PSNR/SSIM/LPIPS tables, runtime benchmark JSON
- `results/figures/` — comparison figures from
  `scripts/generate_figures.py` (Degraded | Restored | Ground Truth,
  identical crops, plus absolute-error panels)
- `docs/data_audit.json` — dataset facts from
  `scripts/audit_dataset.py`
- `docs/experiment_log.md` — ablation tables
- `docs/external_resources.md` — external resource disclosure slide
  content

Do not fabricate any metric, runtime number, or organizer requirement
that is not backed by one of the artifacts above.
