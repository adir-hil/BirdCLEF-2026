# BirdCLEF+ 2026 - Acoustic Species Identification

Kaggle competition solution for identifying 234 wildlife species from audio recordings in Brazil's Pantanal wetlands.

| Detail | Value |
|--------|-------|
| Competition | [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) |
| Task | Multi-label audio classification (234 species) |
| Metric | Macro-averaged ROC-AUC |
| Submission | CPU-only notebook, 90-min runtime, no internet |
| Deadline | June 3, 2026 |

## Approach

- Convert audio to mel spectrograms (128 mel bins, 32kHz sample rate)
- EfficientNet-B0 backbone (timm, pretrained on ImageNet)
- Multi-label classification with BCEWithLogitsLoss
- Train on individual recordings + soundscape annotations
- SpecAugment + mixup + audio augmentations

## Project Structure

```
BirdCLEF-2026/
├── config/
│   └── default.yaml              # Hyperparameters & audio config
├── src/
│   ├── audio.py                  # Audio loading & mel spectrogram
│   ├── dataset.py                # PyTorch datasets
│   ├── model.py                  # EfficientNet-B0 wrapper
│   ├── transforms.py             # Augmentations (SpecAugment, mixup)
│   ├── train.py                  # Training loop
│   ├── evaluate.py               # ROC-AUC evaluation
│   └── utils.py                  # Utilities
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_baseline.ipynb         # Baseline training (Kaggle GPU)
│   └── 03_submission.ipynb       # Inference & submission (Kaggle CPU)
└── scripts/
    └── prepare_data.py           # Data preparation & splits
```

## Setup

```bash
# Clone and install
git clone https://github.com/adir-hil/BirdCLEF-2026.git
cd BirdCLEF-2026
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Download competition data
kaggle competitions download -c birdclef-2026 -p data/
unzip data/birdclef-2026.zip -d data/

# Prepare data splits
python scripts/prepare_data.py
```

## Experiment Log

| Experiment | Model | Epochs | Val ROC-AUC | Notes |
|------------|-------|--------|-------------|-------|
| Baseline | EfficientNet-B0 | 20 | TBD | First submission |
