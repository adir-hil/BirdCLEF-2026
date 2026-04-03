# Next Steps Options

## Current Status: LB Score 0.803 | 1st Place: 0.938 | Gap: 0.135

| # | Option | Expected Impact | Effort | Requires Retraining | Description | Rationale |
|---|--------|----------------|--------|---------------------|-------------|-----------|
| 1 | SED Attention Model | HIGH | Medium | Yes | Switch from simple global-average pooling to Sound Event Detection model with attention pooling. Already built in `src/model.py` (`BirdCLEFSED`). | Attention lets the model focus on the specific time-frequency region where a bird is calling, instead of averaging the whole clip. This is what top BirdCLEF competitors use. |
| 2 | Pseudo-label unlabeled soundscapes | HIGH | Medium | Yes | Use current model to predict labels on 10,592 unlabeled soundscape files, keep high-confidence predictions, retrain on expanded dataset. | Massively expands soundscape training data (from 792 to potentially thousands of windows). Same recording locations as test data — directly closes domain gap. |
| 3 | Lower rating filter | MEDIUM | Low | Yes | Reduce `min_rating` from 3.0 to 1.0 or 2.0. Currently drops 14,254 recordings (40% of data). Optionally weight lower-rated samples less. | More training data with diverse noise conditions improves robustness. Lower-rated recordings still contain valid bird calls — just noisier, which is closer to real soundscape conditions. |
| 4 | Per-species threshold tuning | MEDIUM | Low | No | Optimize a prediction threshold per species on the validation set, instead of using raw sigmoid probabilities. | Some species are systematically over/under-predicted. Per-class calibration can improve ROC-AUC without any model changes. |
| 5 | Ensemble multiple models | HIGH | High | Yes (multiple) | Train 2-3 models (e.g., EfficientNet-B0 + B2, or different folds/seeds) and average predictions at inference. | Ensembles reduce variance and almost always improve scores. Different models make different errors that cancel out when averaged. |

## Recommended Sequence

1. **Option 1** (SED model) — biggest single architectural improvement
2. **Option 3** (lower rating filter) — quick config change, adds 14K samples
3. **Option 2** (pseudo-labeling) — leverages unlabeled soundscape pool
4. **Option 4** (threshold tuning) — free improvement, no retraining
5. **Option 5** (ensemble) — final push if needed
