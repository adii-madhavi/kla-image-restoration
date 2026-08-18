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

## E002 — restoration_candidate (final config, full real dataset)

- Date: 2026-08-18
- Config: configs/final.yaml (commit 1b82f22)
- Model: restoration_candidate (n_features=64, n_groups=4, n_blocks_per_group=6, reduction=16, use_frequency_branch=false)
- Seed: 2026
- Change vs previous experiment: first full run on the real KLA dataset (E001 was the parameter-free bicubic floor)
- Data: real KLA dataset (3200 train pairs -> 2880 train / 320 val, splits/split_seed_2026.json), clip_input=false, geometric aug on, synthetic_degradation aug on (prob 0.3)
- Epochs trained: 150 (full run, no early stopping triggered - loss/PSNR plateaued from ~epoch 115 onward)
- Train loss (final): 0.0723 (composite: charbonnier + 0.15*struct + 0.05*grad)
- Val PSNR / SSIM / LPIPS: 29.02 dB / 0.784 / 0.272 (measured via evaluate.py on the held-out 320-image val split, run.py -> weights/final.pt)
- OOD PSNR / SSIM / LPIPS: not measured (no OOD/hidden set available locally)
- Runtime: not yet benchmarked on target H100 hardware; dev GPU was an NVIDIA RTX 4060 Ti (CUDA), ~71-80s/epoch at batch_size=16
- Params (M): 2.120
- Checkpoint path: weights/final.pt (copied from results/final/checkpoints/best.pt, epoch with best val PSNR)
- Notes / decision: First working end-to-end run after fixing three real bugs surfaced by this training run: (1) Windows DataLoader worker pickling of local closures in train.py/augment.py, (2) synthetic-degradation augmentation could replace a sample with a randomly-sized image and break batch collation (now pinned to the dataset's exact 2x scale contract), (3) SSIMLoss/GradientLoss denominator/epsilon constants underflowed to 0 under fp16 AMP autocast, producing inf loss on every AMP run. run.py verified against the full 400-file Test_NoisyLR set: 0 files failing the output contract (shape/dtype/range/finiteness).

## Preprocessing policy decision

Measured via scripts/audit_dataset.py on the real KLA dataset (docs/data_audit.json):

- Measured %<0 in NoisyLR: mean 0.28% (max 23.4% in the worst image)
- Measured %>1 in NoisyLR: mean 3.11% (max 44.2% in the worst image)
- Decision: do not clip (`clip_input: false` in configs/final.yaml) - a meaningful fraction of real NoisyLR pixels legitimately extend outside [0,1] per the problem statement, and blind-clipping would discard that signal before the model ever sees it.

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
