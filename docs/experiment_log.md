# Experiment Log

Every experiment that could inform the final model/config choice must
be recorded here with enough detail to be reproduced
(Architecture.md §13 "Artifact architecture", Project.md §10 "Winning
philosophy"). Do not report a number in the presentation that is not
traceable to a row in this log.

Template row — copy this block per experiment:

```
## <experiment id, e.g. E007>

- Date:
- Config: configs/<file>.yaml (commit <git sha>)
- Model:
- Seed:
- Change vs previous experiment (one variable only, for ablation claims):
- Data: (real pairs / synthetic augmentation on/off / clip_input policy)
- Epochs trained:
- Train loss (final):
- Val PSNR / SSIM / LPIPS:
- OOD PSNR / SSIM / LPIPS (if measured):
- Runtime: ms/image, batch size, GPU
- Params (M):
- Checkpoint path:
- Notes / decision:
```

## E001 — Baseline (bicubic)

- Date: TBD
- Config: configs/baseline.yaml
- Model: bicubic (parameter-free)
- Seed: 2026
- Change vs previous experiment: N/A (evidence floor)
- Data: TBD
- Epochs trained: N/A
- Train loss (final): N/A
- Val PSNR / SSIM / LPIPS: TBD
- OOD PSNR / SSIM / LPIPS: TBD
- Runtime: TBD
- Params (M): 0
- Checkpoint path: N/A
- Notes / decision: Establishes the floor every learned model must beat.

## Preprocessing policy decision

Record here, once measured (scripts/audit_dataset.py output), whether
`clip_input` was set to True or False for the final config, and why —
Project.md §4 requires this to be an experimentally validated decision,
not a default.

- Measured %<0 in NoisyLR:
- Measured %>1 in NoisyLR:
- Decision: (clip / do not clip) because:

## Loss ablation

| Run | pixel_loss | w_struct | w_grad | Val PSNR | Val SSIM | Val LPIPS |
|-----|-----------|----------|--------|----------|----------|-----------|
| TBD | | | | | | |

## Augmentation ablation

| Run | geometric aug | synthetic degradation | Val PSNR (ID) | Val PSNR (OOD) |
|-----|---------------|------------------------|----------------|-----------------|
| TBD | | | | |

## Model-size / throughput Pareto

| Model | Params (M) | Val PSNR | ms/image (H100) | Notes |
|-------|-----------|----------|-------------------|-------|
| bicubic | 0 | | | |
| residual_sr | | | | |
| restoration_candidate (lightweight) | | | | |
| restoration_candidate (final) | | | | |

## Frequency-branch ablation

| use_frequency_branch | Val PSNR | Val SSIM | Params (M) | ms/image |
|-----------------------|----------|----------|------------|----------|
| False | | | | |
| True | | | | |
