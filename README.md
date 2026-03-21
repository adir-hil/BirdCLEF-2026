# BirdCLEF+ 2026 - Acoustic Species Identification

Kaggle competition solution for identifying 234 wildlife species from audio recordings in Brazil's Pantanal wetlands.

| Detail | Value |
|--------|-------|
| Competition | [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) |
| Task | Multi-label audio classification (234 species) |
| Metric | Macro-averaged ROC-AUC |
| Submission | CPU-only notebook, 90-min runtime, no internet |
| Deadline | June 3, 2026 |

## Data Strategy (16 GB)

**The competition data stays on Kaggle — you never download it locally.**

| What | Where | Why |
|------|-------|-----|
| Competition data (16 GB) | Kaggle `/kaggle/input/birdclef-2026` | Attached as input to notebooks |
| Source code (`src/`, `config/`) | Kaggle dataset `birdclef-src` | Upload this repo as a Kaggle dataset |
| Trained model weights | Kaggle dataset `birdclef-model-weights` | Upload `.pth` files after training |
| This GitHub repo | Local + GitHub | Version control for code |

### Workflow

```
GitHub repo ──upload──> Kaggle Dataset "birdclef-src"
                              │
    ┌─────────────────────────┼────────────────────────┐
    │                         │                        │
    ▼                         ▼                        ▼
02_baseline.ipynb    04_advanced_training.ipynb    01_eda.ipynb
   (Kaggle GPU)          (Kaggle GPU)              (Kaggle GPU)
    │                         │
    ▼                         ▼
 best_model.pth      fold0-4_best.pth
    │                         │
    └──────upload──> Kaggle Dataset "birdclef-model-weights"
                              │
                              ▼
               03_submission.ipynb  OR  05_ensemble_inference.ipynb
                        (Kaggle CPU, no internet, ≤90 min)
                              │
                              ▼
                        submission.csv
```

## Approach

### Baseline (Notebook 02)
- Mel spectrogram (128 mels, 32kHz) → EfficientNet-B0
- BCEWithLogitsLoss, AdamW, cosine annealing
- SpecAugment + mixup + audio augmentations

### Medal Approach (Notebook 04 + 05)
- **SED Architecture**: EfficientNet-B1 backbone with attention pooling
  - Learns *which time frames* contain each species (handles noisy soundscapes)
- **5-Fold CV**: Trains 5 models for robust ensemble
- **Focal Loss**: Handles extreme class imbalance (234 species, most absent)
- **Domain Adaptation**: Includes soundscape data in training to bridge the gap
- **Ensemble**: Mean averaging of fold predictions at inference time

## Project Structure

```
BirdCLEF-2026/
├── config/
│   ├── default.yaml                 # Baseline config
│   └── sed_efficientnet_b1.yaml     # Medal config (SED + focal loss)
├── src/
│   ├── audio.py                     # Audio loading & mel spectrogram
│   ├── dataset.py                   # Datasets + k-fold + balanced sampling
│   ├── model.py                     # BirdCLEFModel, BirdCLEFSED, BirdCLEFModelV2
│   ├── losses.py                    # Focal, Asymmetric, BCE with smoothing
│   ├── transforms.py                # SpecAugment, mixup, CutMix
│   ├── train.py                     # Training loop (standalone)
│   ├── evaluate.py                  # ROC-AUC evaluation
│   └── utils.py                     # Seed, Timer, ensemble utils
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_baseline.ipynb            # Baseline training (Kaggle GPU)
│   ├── 03_submission.ipynb          # Single-model inference (Kaggle CPU)
│   ├── 04_advanced_training.ipynb   # SED + k-fold training (Kaggle GPU)
│   └── 05_ensemble_inference.ipynb  # Ensemble inference (Kaggle CPU)
├── scripts/
│   └── prepare_data.py              # Data preparation & analysis
├── COMPETITION_DETAILS.md           # Full competition reference
└── requirements.txt                 # Python dependencies
```

## Quick Start

### Step 1: Upload source code to Kaggle
1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → New Dataset
2. Name it `birdclef-src`
3. Upload `src/`, `config/`, and `requirements.txt`

### Step 2: Run EDA (optional)
1. Create new Kaggle notebook → Add `birdclef-2026` + `birdclef-src` as inputs
2. Copy contents of `01_eda.ipynb`

### Step 3: Train baseline
1. Create Kaggle notebook with **GPU accelerator**
2. Add inputs: `birdclef-2026` + `birdclef-src`
3. Copy `02_baseline.ipynb` → Run

### Step 4: Train 5-fold SED models
1. Create Kaggle notebook with **GPU accelerator**
2. Add inputs: `birdclef-2026` + `birdclef-src`
3. Copy `04_advanced_training.ipynb`
4. Set `FOLD = 0` → Run → Download weights
5. Repeat for `FOLD = 1, 2, 3, 4`

### Step 5: Upload weights
1. Create Kaggle dataset `birdclef-model-weights`
2. Upload all `*_best.pth` files

### Step 6: Submit
1. Create Kaggle notebook → **CPU, no internet**
2. Add inputs: `birdclef-2026` + `birdclef-model-weights` + `birdclef-src`
3. Copy `05_ensemble_inference.ipynb` → Submit

## Experiment Log

| Experiment | Model | Folds | Loss | Val ROC-AUC | LB Score | Notes |
|------------|-------|-------|------|-------------|----------|-------|
| Baseline | EfficientNet-B0 | 1 | BCE | TBD | TBD | First submission |
| SED v1 | EfficientNet-B1 SED | 5 | Focal | TBD | TBD | + attention pooling |

## Key Technical Insights

1. **Domain gap is the #1 challenge**: Train audio is clean isolated calls; test is noisy field recordings
2. **SED > Classification**: Attention pooling learns to focus on relevant time frames in noisy audio
3. **CPU inference budget**: ~90 min for ~7200 windows → must be efficient (no heavy TTA)
4. **Class imbalance**: 234 species, long-tail distribution → focal loss + balanced sampling
5. **Soundscape training data**: Critical bridge between clean recordings and test distribution
