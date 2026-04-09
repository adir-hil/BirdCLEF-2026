# Next Steps Options

## Current Status: LB Score 0.856 | 1st Place: 0.938 | Gap: 0.082

| # | Option | Expected Impact | Effort | Requires Retraining | Status | Description | Rationale |
|---|--------|----------------|--------|---------------------|--------|-------------|-----------|
| 0 | Scale up EfficientNet (B0 -> B1) | MEDIUM | Low | Yes | DONE (no gain) | Upgrade backbone from B0 (4.3M params) to B1 (6.8M params). | B1 scored 0.802 vs B0's 0.803. Bigger model didn't help — bottleneck is domain gap, not model capacity. |
| 1 | SED Attention Model | HIGH | Medium | Yes | DONE (no gain) | Switch from simple global-average pooling to Sound Event Detection model with attention pooling. Already built in `src/model.py` (`BirdCLEFSED`). | Scored 0.855 vs 0.856. 5-second windows too short for attention. Architecture not the bottleneck. |
| 2 | Pseudo-label unlabeled soundscapes | HIGH | Medium | Yes | IN PROGRESS | Use current model to predict labels on 10,592 unlabeled soundscape files, keep high-confidence predictions, retrain on expanded dataset. | Massively expands soundscape training data (from 792 to potentially thousands of windows). Same recording locations as test data — directly closes domain gap. |
| 3 | Lower rating filter | MEDIUM | Low | Yes | DONE (+0.053) | Remove `min_rating` filter entirely (was 3.0, dropping 40% of data). Use all 35,549 recordings. | Scored 0.856 vs 0.803 (+0.053). Second biggest improvement after soundscape data. Noisy recordings directly close domain gap. |
| 4 | Per-species threshold tuning | MEDIUM | Low | No | Pending | Optimize a prediction threshold per species on the validation set, instead of using raw sigmoid probabilities. | Some species are systematically over/under-predicted. Per-class calibration can improve ROC-AUC without any model changes. |
| 5 | Ensemble multiple models | HIGH | High | Yes (multiple) | Pending | Train 2-3 models (e.g., EfficientNet-B0 + B2, or different folds/seeds) and average predictions at inference. | Ensembles reduce variance and almost always improve scores. Different models make different errors that cancel out when averaged. |

## Recommended Sequence

1. ~~**Option 0** (scale up B0 -> B1) — no improvement, reverted~~
2. ~~**Option 3** (lower rating filter) — scored 0.856 (+0.053), DONE~~
3. ~~**Option 1** (SED model) — scored 0.855, no improvement~~
4. **Option 2** (pseudo-labeling) — leverages unlabeled soundscape pool **(NEXT)**
5. **Option 4** (threshold tuning) — free improvement, no retraining
6. **Option 5** (ensemble) — final push if needed
