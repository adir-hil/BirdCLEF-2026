# Training Insights & Analysis

## Model Comparison Table

| Metric | v2 (B0, rating≥3) | v3 (B1, rating≥3) | v4 (B0, all ratings) |
|--------|-------------------|-------------------|----------------------|
| Backbone | EfficientNet-B0 | EfficientNet-B1 | EfficientNet-B0 |
| Parameters | 4.3M | 6.8M | 4.3M |
| Train samples | 17,038 | 17,038 | 28,440 |
| Val samples | 4,257 | 4,257 | 7,109 |
| Best Val AUC | 0.9865 | 0.9767 | 0.9797 |
| Best epoch | 18 | 10 | 10 |
| Final loss | 0.0126 | 0.0148 | 0.0125 |
| Stopped at | Epoch 18 | Epoch 13 | Epoch 13 |
| LB Score | 0.803 | 0.802 | TBD |

---

## Insight #1: Val AUC is not a reliable proxy for LB score

The validation set is composed of clean recordings (same distribution as training). The test set is real-world soundscapes — noisy, continuous, multi-species. The two distributions are fundamentally different. A model that scores 0.9865 on clean val can score only 0.803 on noisy soundscapes. **The LB score is the only reliable measure of progress.**

---

## Insight #2: Scaling up the model (B0 → B1) doesn't help

B1 (6.8M params) scored 0.802 vs B0's 0.803. The bottleneck is not model capacity — it's the **domain gap** between clean training recordings and noisy test soundscapes. Adding more parameters without addressing the domain gap provides no benefit.

**Lesson:** When the domain gap is the main bottleneck, focus on data and architecture choices that bridge that gap (soundscape training data, SED attention model), not on scaling up the backbone.

---

## Insight #3: Lower val AUC with more data is not necessarily bad

v4 (all ratings, no filter) shows val AUC of 0.9797 vs v2's 0.9865 — seemingly worse. However:
- The **val set itself changed** — now contains noisier recordings (rating < 3), making it harder
- The **loss is lowest** of all runs (0.0125) — the model is learning better overall patterns
- The comparison is **not apples-to-apples** — v2's val was only clean recordings, v4's val contains noise

The v4 model may generalize better to soundscapes despite the lower val number. **The LB score will tell the real story.**

---

## Insight #4: More data leads to faster convergence

Both B1 (v3) and B0-all-ratings (v4) peak at epoch 10, vs epoch 18 for v2. More diverse training data provides richer gradient signals, allowing the model to find good weights earlier. Early stopping activates sooner, which also reduces overfitting risk.

---

## Insight #5: The domain gap is the primary challenge

| Source | LB Score | Delta |
|--------|----------|-------|
| Clean recordings only | 0.723 | baseline |
| + 792 soundscape windows | 0.797 | +0.074 |
| + TTA x3 | 0.803 | +0.006 |
| B1 backbone | 0.802 | -0.001 |

The single biggest jump came from adding soundscape training data (+0.074). Everything else has been incremental. **Closing the domain gap further (pseudo-labeling, SED model) is the highest-priority direction.**

---

## Open Questions

- Will v4 (all ratings) improve the LB score despite lower val AUC? *(pending submission)*
- How much will the SED attention model help over simple global-average pooling?
- Can pseudo-labeling 10,592 unlabeled soundscapes further close the domain gap?
