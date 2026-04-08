# Training Insights & Analysis

## Model Comparison Table

| Metric | v2 (B0, rating≥3) | v3 (B1, rating≥3) | v4 (B0, all ratings) | v5 (SED, all ratings) |
|--------|-------------------|-------------------|----------------------|----------------------|
| Backbone | EfficientNet-B0 | EfficientNet-B1 | EfficientNet-B0 | EfficientNet-B0 (features_only) |
| Parameters | 4.3M | 6.8M | 4.3M | 3.85M |
| Train samples | 17,038 | 17,038 | 28,440 | 28,440 |
| Val samples | 4,257 | 4,257 | 7,109 | 7,109 |
| Best Val AUC | 0.9865 | 0.9767 | 0.9797 | 0.9707 (fresh) |
| Best epoch | 18 | 10 | 10 | 8 |
| Final loss | 0.0126 | 0.0148 | 0.0125 | 0.0144 |
| Stopped at | Epoch 18 | Epoch 13 | Epoch 13 | Epoch 11 |
| LB Score | 0.803 | 0.802 | **0.856** | 0.855 |

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
| All ratings (min_rating=0) | 0.856 | +0.053 |
| SED attention model | 0.855 | -0.001 |

The two biggest jumps came from domain-gap interventions: adding soundscape training data (+0.074) and using all ratings including noisy recordings (+0.053). Architectural changes (B1 scaling, SED attention) had no effect. **The bottleneck is data and domain gap, not model architecture. Pseudo-labeling and data-centric approaches are the highest-priority direction.**

---

---

## Insight #6: Noisier training data directly closes the domain gap

Removing the `min_rating` filter (using all ratings, 35,549 recordings vs 21,295) produced +0.053 improvement — the second biggest gain in the project. This confirms that the model benefits from training on noisy, low-quality recordings that more closely resemble real-world soundscape conditions at test time.

**Lesson:** Curating for "clean" training data can hurt generalization when the test distribution is noisy. For soundscape tasks, more data with diverse noise conditions beats fewer clean examples.

---

---

## Insight #7: SED attention model doesn't help on 5-second windows

Replacing global average pooling with learned attention-based pooling (BirdCLEFSED) scored 0.855 vs the simple B0's 0.856. This is the third architectural change that failed to improve the score (after B1 scaling and now SED).

**Why it didn't help:** SED attention learns which time frames matter per species — critical for long recordings where a bird calls briefly. But with 5-second windows, there isn't enough temporal variation for attention to exploit. Global average pooling is already effective at this short duration. The SED model also had fewer parameters (3.85M vs 4.3M) due to `features_only=True` dropping the backbone's final classification layers.

**Key pattern:** Three architectural experiments (B1, SED, all on same data) produced scores of 0.802, 0.855, 0.856. All within noise. Meanwhile, two data experiments (soundscapes +0.074, all ratings +0.053) produced large gains. **Architecture is not the bottleneck.**

---

## Open Questions

- Can pseudo-labeling 10,592 unlabeled soundscapes further close the domain gap?
- Would longer windows (10s or 15s) make SED attention more effective?
