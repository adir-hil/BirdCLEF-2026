# Submission History

## BirdCLEF+ 2026 — Kaggle Competition

| # | Date | Description | LB Score | LB Rank | Model Dataset | Architecture | Inference | Preprocessing | Train Data | Augmentation | Batch Size | LR | Optimizer | Scheduler | Epochs (actual) | Early Stop | Loss | Val AUC | Train/Val Split |
|---|------|-------------|----------|---------|---------------|--------------|-----------|---------------|------------|--------------|------------|-----|-----------|-----------|-----------------|------------|------|---------|-----------------|
| 1 | 2026-03-28 | Baseline v1 — EfficientNet-B0 on clean recordings only | 0.723 | 1182 | birdclef-baseline-model | EfficientNet-B0 (4.3M params) | Single pass | Mel spectrogram (128 mels, 32kHz, 5s windows) | 17,038 clean recordings | SpecAugment + Mixup (alpha=0.5) + audiomentations (noise, pitch, gain, shift) | 64 | 0.001 | AdamW (wd=0.01) | Cosine + 2-epoch warmup | 18 (early stop at 15+3) | patience=3 | BCEWithLogitsLoss | 0.9865 | 80/20 stratified (min_rating=3.0) |
| 2 | 2026-04-03 | Added 66 labeled soundscape files with stratified sampling (792 windows) | 0.797 | — | birdclef-baseline-model-v2 | EfficientNet-B0 (4.3M params) | Single pass | Mel spectrogram (128 mels, 32kHz, 5s windows) | 17,830 (17,038 clean + 792 soundscape) | SpecAugment + Mixup (alpha=0.5) + audiomentations | 64 | 0.001 | AdamW (wd=0.01) | Cosine + 2-epoch warmup | 18 (early stop) | patience=3 | BCEWithLogitsLoss | ~0.98 | 80/20 stratified (min_rating=3.0) |
| 3 | 2026-04-03 | Added TTA at inference (3 variants: original, time-shift +0.5s, time-reverse) | 0.803 | — | birdclef-baseline-model-v2 | EfficientNet-B0 (4.3M params) | TTA x3 (avg) | Mel spectrogram (128 mels, 32kHz, 5s windows) | 17,830 (17,038 clean + 792 soundscape) | SpecAugment + Mixup (alpha=0.5) + audiomentations | 64 | 0.001 | AdamW (wd=0.01) | Cosine + 2-epoch warmup | 18 (early stop) | patience=3 | BCEWithLogitsLoss | ~0.98 | 80/20 stratified (min_rating=3.0) |
| 4 | 2026-04-04 | Upgraded backbone to EfficientNet-B1 (6.8M params) | 0.802 | — | birdclef-baseline-model-v3 | EfficientNet-B1 (6.8M params) | TTA x3 (avg) | Mel spectrogram (128 mels, 32kHz, 5s windows) | 17,830 (17,038 clean + 792 soundscape) | SpecAugment + Mixup (alpha=0.5) + audiomentations | 64 | 0.001 | AdamW (wd=0.01) | Cosine + 2-epoch warmup | 13 (early stop at 10+3) | patience=3 | BCEWithLogitsLoss | 0.9767 | 80/20 stratified (min_rating=3.0) |

## Key Changes Between Submissions

- **#1 -> #2 (+0.074):** Trained on soundscape data to bridge domain gap. Parsed HH:MM:SS label format, filtered to 66 labeled files, stratified sampling (all positives + 3x negatives). 28 species existed ONLY in soundscapes.
- **#2 -> #3 (+0.006):** Test-time augmentation with 3 variants per window. Same model, no retraining needed. Audio loaded once, augmentations via fast numpy ops.
- **#3 -> #4 (-0.001):** Scaled up backbone from B0 to B1. No improvement — B1 early-stopped at epoch 13 with lower val AUC (0.9767 vs ~0.98). Bigger model did not help; may need more data or different architecture rather than more parameters.

## Competition Target

- 1st place score: 0.938
- Current best: 0.803 (submission #3, EfficientNet-B0 + TTA)
- Current gap to 1st: 0.135
