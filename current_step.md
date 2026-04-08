# Current Step: Pseudo-Label Unlabeled Soundscapes (Option 2)

## Overview

Use the best model (submission #5, EfficientNet-B0, LB 0.856) to predict labels on ~10,592 unlabeled soundscape files. Keep high-confidence predictions as pseudo-labels, add them to the training set, and retrain.

## Why This Should Help

1. **Massive data expansion** — from 66 labeled soundscape files to potentially thousands
2. **Same domain as test data** — soundscapes are recorded at the same locations/conditions as test data
3. **Directly targets the bottleneck** — every improvement so far came from data changes, not architecture
4. **28 species only exist in soundscapes** — more soundscape data = better coverage of these species

## Implementation

### Step 1: Generate pseudo-labels (notebook `06_pseudo_labels.ipynb`)
- Run on Kaggle GPU with datasets: `birdclef-2026`, `birdclef-baseline-model-v4`, `birdclef-src`
- Loads best model, runs inference on all unlabeled soundscapes
- Keeps predictions with confidence >= 0.5
- Saves `pseudo_labels.csv` and `combined_soundscape_labels.csv`

### Step 2: Upload pseudo-labels
- Download output CSV from notebook
- Upload as Kaggle dataset `birdclef-pseudo-labels`

### Step 3: Retrain with pseudo-labels
- In training notebook, load `combined_soundscape_labels.csv` instead of `train_soundscapes_labels.csv`
- Everything else stays the same (B0, all ratings, TTA x3)

### Step 4: Submit and compare
- Upload trained model as `birdclef-baseline-model-v6`
- Run submission notebook
- Compare against 0.856 baseline

## Risk Assessment

- **Medium risk**: Pseudo-labels may contain errors that poison the training
- **Mitigation**: Using 0.5 threshold to keep only confident predictions
- **If it degrades**: Can try higher threshold (0.7) or weight pseudo-labels lower
