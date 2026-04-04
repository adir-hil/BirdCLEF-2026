# Current Step: Lower Rating Filter (Use All Training Recordings)

## Overview

Remove the `min_rating >= 3.0` filter that was dropping 14,254 recordings (40% of the data). This gives the model more diverse training examples, including noisier recordings that better resemble real-world soundscape conditions.

## What Changed

In `config/default.yaml`:
```yaml
# Before
min_rating: 3.0   # drops 14,254 recordings

# After
min_rating: 0     # keeps all 35,549 recordings
```

Also reverted backbone to EfficientNet-B0 (B1 showed no improvement in submission #4).

## Why This Should Help

1. **More data** — 35,549 recordings instead of 21,295 (+67% more training samples)
2. **Noisier = better** — lower-rated recordings contain background noise, overlapping species, and poor recording conditions. This is closer to what the model sees in real soundscapes at test time.
3. **More species coverage** — some species may only have low-rated recordings. Dropping them means zero training examples for those species.
4. **Domain gap reduction** — the main bottleneck (0.803 vs 0.938) is the model struggling on noisy real-world audio. Training on noisy data directly addresses this.

## Expected Training Data

| Dataset | Before (min_rating=3.0) | After (min_rating=0) |
|---------|------------------------|---------------------|
| Clean recordings | 17,038 train + 4,257 val | ~28,439 train + ~7,110 val |
| Soundscape windows | 792 | 792 |
| **Total train** | **17,830** | **~29,231** |

## What Stays the Same

- Architecture: EfficientNet-B0 (4.3M params)
- Preprocessing: Mel spectrogram (128 mels, 32kHz, 5s windows)
- Augmentation: SpecAugment + Mixup + audiomentations
- Optimizer: AdamW (lr=0.001, wd=0.01)
- Scheduler: Cosine + 2-epoch warmup
- Loss: BCEWithLogitsLoss
- Epochs: 18 max, early stopping patience=3
- Inference: TTA x3

## Steps to Execute

1. **Update `birdclef-src` dataset on Kaggle** — ZIP latest `src/`, `config/`, `requirements.txt`, upload as new version
2. **Download `02_baseline.ipynb`** from GitHub and upload to Kaggle
3. **Run training** — Save & Run All (internet ON, GPU)
4. **Save trained model** as `birdclef-baseline-model-v4`
5. **Update `03_submission.ipynb`** model path to `birdclef-baseline-model-v4`
6. **Submit** — Save & Run All (internet OFF)
7. **Record results** in `submissions.md`

## Risk Assessment

- **Low risk**: Simple config change, no code modifications
- **Possible issue**: More data = longer training (~8-10 hours instead of ~4-5). May hit Kaggle's 12h GPU limit, but with early stopping should be fine.
- **Possible issue**: Lower-quality recordings may add noise to training. If LB degrades, can try intermediate values (min_rating=1.0 or 2.0).
