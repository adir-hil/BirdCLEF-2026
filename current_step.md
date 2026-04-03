# Current Step: Scale Up EfficientNet-B0 to B1

## Overview

Upgrade the model backbone from EfficientNet-B0 (4.3M parameters) to EfficientNet-B1 (7.8M parameters) to increase model capacity without changing the training pipeline or architecture type.

## What Changed

A single line in `config/default.yaml`:
```yaml
# Before
backbone: tf_efficientnet_b0_ns

# After
backbone: tf_efficientnet_b1_ns
```

Both use Noisy Student (`_ns`) pretrained weights from ImageNet, loaded via the `timm` library.

## Why B1

| Property | B0 | B1 | B2 |
|----------|-----|-----|-----|
| Parameters | 4.3M | 7.8M | 9.1M |
| Default resolution | 224 | 240 | 260 |
| Relative compute | 1x | ~1.5x | ~2x |

- **B0** is the smallest EfficientNet — fast but limited capacity. Our current LB score (0.803) may be partly limited by model size.
- **B1** offers ~1.8x more parameters with moderate compute increase. Should still fit within the 90-minute CPU inference budget (with TTA x3).
- **B2** was considered but may be too slow for CPU inference with TTA. Can be tested later if B1 fits comfortably.

## What Stays the Same

Everything except the backbone size:
- Training data: 17,830 samples (17,038 clean + 792 soundscape windows from 66 labeled files)
- Preprocessing: Mel spectrogram (128 mels, 32kHz, 5-second windows)
- Augmentation: SpecAugment + Mixup (alpha=0.5) + audiomentations
- Optimizer: AdamW (lr=0.001, weight_decay=0.01)
- Scheduler: Cosine annealing + 2-epoch warmup
- Loss: BCEWithLogitsLoss
- Epochs: 18 max, early stopping patience=3
- Batch size: 64
- Inference: TTA x3 (original, time-shift +0.5s, time-reverse)

## Expected Outcome

- Slightly higher validation AUC due to increased model capacity
- Potential LB improvement of +0.01 to +0.03 (estimated)
- Training time may increase by ~30-50% (more parameters to update per batch)
- Inference time may increase by ~30-50% per forward pass (must verify it stays within 90 min)

## Steps to Execute

1. **Update `birdclef-src` dataset on Kaggle** — ZIP latest `src/`, `config/`, `requirements.txt` and upload as new version
2. **Download `02_baseline.ipynb`** from GitHub and upload to Kaggle as training notebook
3. **Run training** — Save & Run All (internet ON, GPU enabled)
4. **Save trained model** as new Kaggle dataset: `birdclef-baseline-model-v3`
5. **Update inference notebook** (`03_submission.ipynb`) to point to `birdclef-baseline-model-v3`
6. **Submit inference notebook** — Save & Run All (internet OFF)
7. **Record results** in `submissions.md`

## Risk Assessment

- **Low risk**: Drop-in replacement, no code changes needed beyond config
- **Main concern**: CPU inference time with B1 + TTA x3. If too slow, can reduce TTA from 3 to 2 variants
- **Fallback**: If B1 doesn't improve LB score, revert config to B0 and proceed to Option 1 (SED model)
