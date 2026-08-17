# weights/

Place the final trained checkpoint here as `weights/final.pt`
(the default path `inference.py` and `benchmark.py` load — see
`configs/final.yaml`).

## Format

Produced by `src/engine/checkpoint.py::save_checkpoint`, a
`torch.save`d dict containing:

```
model_state_dict
optimizer_state_dict   # training checkpoints only; not required for inference
scheduler_state_dict   # training checkpoints only; not required for inference
epoch
metrics
config                  # the resolved training config
seed
git_commit
```

`inference.py` only reads `model_state_dict`.

## Obtaining the weights

- If training from scratch: `python train.py --config configs/final.yaml`
  writes checkpoints to `results/final/checkpoints/{best,last}.pt`; copy
  the chosen checkpoint to `weights/final.pt`.
- If the checkpoint exceeds normal Git storage limits, host it via an
  accessible download method permitted by the portal (e.g. a release
  asset or Git LFS) and document the download URL and checksum here:

```
weights/final.pt
  sha256: <fill in>
  download: <fill in>
```

## Do not commit

- Intermediate/experimental checkpoints (keep those out of git; log
  their results in `docs/experiment_log.md` instead).
- Optimizer states for the final submission artifact are optional —
  training-reproducibility only requires the training *code* + config
  + seed to be sufficient to reproduce the checkpoint
  (Submission_and_Repository.md §4), not that the optimizer state
  itself ships.
