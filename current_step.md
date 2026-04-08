# Current Step: SED Attention Model — DONE ✓ | LB: 0.855 (no improvement)

## Result

SED attention model scored 0.855 vs the simple B0's 0.856 — no meaningful improvement. This is the third architectural change that failed (B1: 0.802, SED: 0.855). The bottleneck is data and domain gap, not model architecture.

## What Was Tried

- Switched `model_type` from `simple` to `sed` in config
- BirdCLEFSED: EfficientNet-B0 backbone with `features_only=True` + frequency pooling + learned attention over time frames
- 3.85M params (vs 4.3M for simple B0)
- Training was messy: notebook had duplicate training cells, model trained for 11+6 epochs

## Key Takeaway

Architecture changes have consistently failed to improve the score. Data changes (soundscapes +0.074, all ratings +0.053) are the only things that worked. Next step should be data-centric: pseudo-labeling unlabeled soundscapes.

## Next Step: Option 2 — Pseudo-label unlabeled soundscapes

Use the current best model (simple B0, submission #5) to predict labels on 10,592 unlabeled soundscape files, keep high-confidence predictions, and retrain on the expanded dataset.
