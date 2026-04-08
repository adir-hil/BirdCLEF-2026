# Submission History

## BirdCLEF+ 2026 — Kaggle Competition

| # | Date | Description | LB Score | LB Rank | Model Dataset | Architecture | Inference | Preprocessing | Train Data | Augmentation | Batch Size | LR | Optimizer | Scheduler | Epochs (actual) | Early Stop | Loss | Val AUC | Train/Val Split |
|---|------|-------------|----------|---------|---------------|--------------|-----------|---------------|------------|--------------|------------|-----|-----------|-----------|-----------------|------------|------|---------|-----------------|
| 1 | 2026-03-28 | Baseline v1 — EfficientNet-B0 on clean recordings only | 0.723 | 1182 | birdclef-baseline-model | EfficientNet-B0 (4.3M params) | Single pass | Mel spectrogram (128 mels, 32kHz, 5s windows) | 17,038 clean recordings | SpecAugment + Mixup (alpha=0.5) + audiomentations (noise, pitch, gain, shift) | 64 | 0.001 | AdamW (wd=0.01) | Cosine + 2-epoch warmup | 18 (early stop at 15+3) | patience=3 | BCEWithLogitsLoss | 0.9865 | 80/20 stratified (min_rating=3.0) |
| 2 | 2026-04-03 | Added 66 labeled soundscape files with stratified sampling (792 windows) | 0.797 | — | birdclef-baseline-model-v2 | EfficientNet-B0 (4.3M params) | Single pass | Mel spectrogram (128 mels, 32kHz, 5s windows) | 17,830 (17,038 clean + 792 soundscape) | SpecAugment + Mixup (alpha=0.5) + audiomentations | 64 | 0.001 | AdamW (wd=0.01) | Cosine + 2-epoch warmup | 18 (early stop) | patience=3 | BCEWithLogitsLoss | ~0.98 | 80/20 stratified (min_rating=3.0) |
| 3 | 2026-04-03 | Added TTA at inference (3 variants: original, time-shift +0.5s, time-reverse) | 0.803 | — | birdclef-baseline-model-v2 | EfficientNet-B0 (4.3M params) | TTA x3 (avg) | Mel spectrogram (128 mels, 32kHz, 5s windows) | 17,830 (17,038 clean + 792 soundscape) | SpecAugment + Mixup (alpha=0.5) + audiomentations | 64 | 0.001 | AdamW (wd=0.01) | Cosine + 2-epoch warmup | 18 (early stop) | patience=3 | BCEWithLogitsLoss | ~0.98 | 80/20 stratified (min_rating=3.0) |
| 4 | 2026-04-04 | Upgraded backbone to EfficientNet-B1 (6.8M params) | 0.802 | — | birdclef-baseline-model-v3 | EfficientNet-B1 (6.8M params) | TTA x3 (avg) | Mel spectrogram (128 mels, 32kHz, 5s windows) | 17,830 (17,038 clean + 792 soundscape) | SpecAugment + Mixup (alpha=0.5) + audiomentations | 64 | 0.001 | AdamW (wd=0.01) | Cosine + 2-epoch warmup | 13 (early stop at 10+3) | patience=3 | BCEWithLogitsLoss | 0.9767 | 80/20 stratified (min_rating=3.0) |
| 5 | 2026-04-05 | Removed rating filter — trained on all 35,549 recordings (min_rating=0) | 0.856 | — | birdclef-baseline-model-v4 | EfficientNet-B0 (4.3M params) | TTA x3 (avg) | Mel spectrogram (128 mels, 32kHz, 5s windows) | 29,232 (28,440 clean + 792 soundscape) | SpecAugment + Mixup (alpha=0.5) + audiomentations | 32 | 0.001 | AdamW (wd=0.01) | Cosine + 2-epoch warmup | 13 (early stop at 10+3) | patience=3 | BCEWithLogitsLoss | 0.9797 | 80/20 stratified (min_rating=0) |
| 6 | 2026-04-08 | SED attention model — replaced global avg pooling with learned attention | 0.855 | — | birdclef-baseline-model-v5 | BirdCLEFSED (B0 backbone, 3.85M params) | TTA x3 (avg) | Mel spectrogram (128 mels, 32kHz, 5s windows) | 29,232 (28,440 clean + 792 soundscape) | SpecAugment + Mixup (alpha=0.5) + audiomentations | 32 | 0.001 | AdamW (wd=0.01) | Cosine + 2-epoch warmup | 11+6 (double run, see notes) | patience=3 | BCEWithLogitsLoss | 0.9809 (continued) / 0.9707 (fresh) | 80/20 stratified (min_rating=0) |

## Key Changes Between Submissions

- **#1 -> #2 (+0.074):** Trained on soundscape data to bridge domain gap. Parsed HH:MM:SS label format, filtered to 66 labeled files, stratified sampling (all positives + 3x negatives). 28 species existed ONLY in soundscapes.
- **#2 -> #3 (+0.006):** Test-time augmentation with 3 variants per window. Same model, no retraining needed. Audio loaded once, augmentations via fast numpy ops.
- **#3 -> #4 (-0.001):** Scaled up backbone from B0 to B1. No improvement — B1 early-stopped at epoch 13 with lower val AUC (0.9767 vs ~0.98). Bigger model did not help; may need more data or different architecture rather than more parameters.
- **#4 -> #5 (+0.054):** Removed `min_rating` filter entirely (was 3.0), using all 35,549 recordings (+67% data). More diverse, noisier training data directly reduced the domain gap. Second biggest improvement after adding soundscape data.
- **#5 -> #6 (-0.001):** SED attention model with `features_only=True` backbone and learned time-frame attention. No improvement — 5-second windows are too short for attention to help. The bottleneck remains data/domain gap, not architecture. Also had fewer params (3.85M vs 4.3M) and messy double-training run.

## Competition Target

- 1st place score: 0.938
- Current best: 0.856 (submission #5, EfficientNet-B0 + all ratings + TTA)
- Current gap to 1st: 0.082
