# Wunder Challenge: market state prediction

My approaches to the [Wunder Fund](https://wunderfund.io) RNN challenge (Sep–Nov 2025).

## Problem

- 517 independent sequences, each 1000 steps long, with 32 anonymized market features per step.
- Steps 0–99 are warm-up. For steps 100–998, predict the next state vector, one step at a time.
- Metric: R² averaged over the 32 features.
- Inference is online: `solution.py` receives one `DataPoint` at a time (see `utils.py`).

**Result: 444th of ~5,000 (top ~9%), best 0.31990 R² (#9). Top score on the board: 0.39196.**

## Data analysis

Reproduce with `python data_analysis.py` (needs `train.parquet` in the repo root). Findings:

- **Clean, standardized data.** No NaNs. Every feature has mean ≈ 0 and std ≈ 1 (0.985–1.011), and the distribution is stable over time (std 1.01 / 0.99 / 0.99 over steps 0–100 / 100–500 / 500–999). No drift to model.
- **Strongly mean-reverting, weakly autocorrelated.** Lag-1 autocorrelation of the levels is 0.30 (median; range 0.03–0.52 across features).
- **Persistence loses badly.** Predicting `x[t+1] = x[t]` gives R² = −0.39, in line with theory: with lag-1 correlation ρ the persistence R² is 2ρ − 1 ≈ −0.4. The task is predicting reversion, not continuation.
- **Differences are over-differenced.** Lag-1 autocorrelation of first differences is −0.54 (all 32 features between −0.44 and −0.63). That is the signature of differencing a near-stationary series (white noise gives −0.5). Differencing adds MA(1) structure instead of removing a trend. Hypothesis for why every differenced submission (#13–19, #22) lost to raw states.
- **Short memory.** Difference autocorrelation is 0.10 at lag 2 and about 0 at lags 5 and 10 (−0.05, 0.03). Little signal beyond the last few steps.
- **Features are correlated and low-dimensional.** Mean absolute correlation between features is 0.36 (max 0.95). The top 3 principal components explain 73% of the variance and the top 10 explain 98%. Cross-feature structure carries predictive signal.
- **Sequence-level offsets are large.** The spread of per-sequence means is 0.57 of the within-sequence std. This motivated the mean-centering variants (#24–30), which did not help on the leaderboard.

### Baselines

Ridge regression on the last *k* states (all 32 features), trained and validated on a split by `seq_ix` (80/20, seed 29), predicting steps 100–998:

| Model | R² |
|---|---|
| Persistence (`x[t]`) | −0.48 (val split), −0.39 (all data) |
| Ridge, k = 1 | 0.200 |
| Ridge, k = 5 | 0.269 |
| Ridge, k = 20 | 0.285 |
| **My best LSTM-attention (#9, leaderboard)** | **0.320** |
| Top of the leaderboard | 0.392 |

My own split and the hidden test set differ, so the ridge and leaderboard figures are not strictly comparable. Even so, the 31 submissions improve on a 20-step linear model by only about 0.03. The gap from linear to the top score (about 0.11) is larger than the gap from my best to the top (0.07).

## Approaches

Rough chronological order. Scores are still to be added.

### 1. Seq2seq LSTM autoencoder: `first_try.ipynb`, `sequentialKeras.ipynb`
Encode the 100 warm-up steps into a latent vector, then decode all 899 future steps in one shot using `RepeatVector` and an LSTM. MinMax scaling. Never sees new observations after warm-up, so it can't track the sequence.

### 2. Sliding-window one-step LSTM: `SimpleModel.ipynb`, `Untitled-1.ipynb`
Switch to next-step prediction: a 20-step window goes through 2 stacked LSTMs and a Dense layer to predict the next state. `Untitled-1` adds additive attention over the short-term window.

### 3. Stacked dense autoencoders: `Untitled-3.ipynb`
Greedy layer-wise dense autoencoders on 49-step windows, as a feature extractor. Abandoned early.

### 4. Pretrained LSTM encoder + attention-pooling head: `Untitled-2.ipynb`, `Hybrid_LSTM.ipynb`
- Pretrain an LSTM encoder-decoder (16 units) on full 1000-step sequences (`ENCODER_Only_PretrainedLSTM.weights.h5`).
- Separately, a 49-step window goes through 2 LSTMs, learned attention pooling (Dense(tanh), Dense, softmax) and Dense layers to predict the next step.
- Encoder contexts were cached in `precomputed_*_contexts.npy`, which are not committed.

### 5. LSTM encoder-decoder with attention (main line): `AE_Attention.ipynb`, `model.py`
Used for most submissions (1–12, 21, 24–30).
- Encoder: 3 stacked LSTMs (40, 35, 30) with a projected skip connection, plus multi-head self-attention.
- Decoder: LSTM initialized from encoder states, plus multi-head cross-attention over encoder outputs.
- Trained with teacher forcing on 100 → 899 steps.
- Inference: re-encode the last 100 observed states every *N* steps. Between re-encodes, step the decoder fed with the true last state.
- Variations across submissions: re-encode interval *N* (1, 7, 10, 15), subtracting the window mean before encoding, retrained weights, layer sizes.

### 6. Same model on first differences: `Differenced_AE_Attention.ipynb`
Used for submissions 13–19, 22 and `Sub22/made_ct_1`.
- Train on `x[t+1] − x[t]` instead of raw states; weights in `AE_wts_diffs.weights.h5`. Differencing is in the copies under `Submission13/` and `Submission14/`.
- The later top-level notebook drops differencing and adds a causal `Conv1D` front-end, `LayerNormalization`, and wider LSTMs (64, 50, 32).

### 7. iTransformer-style model: `iTransformer.ipynb`
- A 100-step window goes through an LSTM, then is transposed so each *feature* becomes a token.
- Multi-head attention runs across features, followed by Add & Norm and a feed-forward block, to predict the next step.
- Last experiment. Not submitted.

## Layout

- `model.py`, `utils.py`: current model definition and the official scorer.
- `Submission*/`, `Sub*/`, `sub*/`: frozen snapshot of each submission (`solution.py`, `model.py`, weights, training notebook).
- `<folder>/<zip name>/`: what was actually uploaded, where it differs from the folder copy.
- `wnn_starterpack.zip`: official starter pack.

`train.parquet` is not committed. Get it from the starter pack and place it in the repo root.

## Results

Leaderboard R². "Re-encode every N" means the encoder re-reads the last 100 states every N steps. Encoder sizes are the LSTM widths in `model.py`.

| # | Date | Approach | Change | Score |
|---|---|---|---|---|
| 1 | 10-23 | 5 | 2-layer encoder [48, 12], re-encode every 10 | 0.29312 |
| 2 | 10-23 | 5 | Retrained weights (`try2`), re-encode every step | 0.22511 |
| 3 | 10-24 | 5 | 3-layer encoder [48, 20, 12], every 10 | 0.30310 |
| 4 | 10-24 | 5 | Same, every step | 0.22704 |
| 5 | 10-24 | 5 | [48, 20, 24], every 7 | 0.30498 |
| 6 | 10-24 | 5 | Retrained | 0.31194 |
| 7 | 10-24 | 5 | Retrained | 0.29649 |
| 8 | 10-26 | 5 | Retrained | 0.30329 |
| **9** | 11-03 | 5 | [48, 20, 40], every 7 | **0.31990** |
| 10 | 11-03 | 5 | [24, 20, 40] | 0.31700 |
| 11 | 11-03 | 5 | Same, every step | 0.27771 |
| 12 | 11-04 | 5 | [48, 20, 20] | 0.31119 |
| 13 | 11-06 | 6 | Weights trained on differences [40, 40, 40], but inference fed raw states | 0.22792 |
| 14 | 11-06 | 6 | Inference fed differences | −0.31231 |
| 15 | 11-06 | 6 | Retrained, decoder fed differences | 0.05357 |
| 16 | 11-06 | 6 | Retrained | 0.05357 |
| 17 | 11-06 | 6 | Retrained | 0.05357 |
| 18 | 11-11 | 6 | Added Conv1D front-end | −0.51097 |
| 19 | 11-11 | 6 | Conv1D, retrained | −365.51116 |
| try | 11-12 | ? | Not in repo | −0.31231 |
| 21 | 11-12 | 5 | Conv1D + [48, 40, 40], every 7 | 0.31544 |
| 22 | 11-12 | 6 | Differences, every 7 | 0.21829 |
| `Sub22/made_ct_1` | 11-12 | 6 | Differences, every step | 0.09825 |
| 24 | 11-14 | 5 | [40, 40, 40], subtract window mean, every 7 | 0.26987 |
| 25 | 11-14 | 5 | Mean-centering variant | 0.14806 |
| `Sub24/made_ct_1` | 11-14 | 5 | Mean-centered, every step | 0.12908 |
| 27 | 11-16 | 5 | Mean-centered, every step | 0.22151 |
| 28 | 11-16 | 5 | Retrained | 0.29039 |
| 29 | 11-16 | 5 | [40, 35, 30] | 0.29444 |
| 30 | 11-16 | 5 | Every 15 (one upload failed the check; re-upload scored) | 0.30158 |

### Takeaways

- **Best: #9 (0.31990).** Plain encoder-decoder with attention, re-encoding every 7 steps.
- **Re-encoding every step always hurt** (#2, #4, #11, #27, both `made_ct_1`), by 0.02–0.12 against the matching every-7 or every-10 run.
- **Differencing never beat raw states.** The best differenced run was #13, and that one fed raw states at inference anyway.
- **#15–17 scored exactly 0.05357 despite different weights.** That suggests the prediction ignored the weights, i.e. an inference bug.
- **Conv1D on the differenced model failed:** #18 scored −0.51, and #19 diverged to −365 (an autoregressive blow-up). The same Conv1D on raw states (#21) scored 0.31544.
- **Mean-centering (#24–30) never beat the earlier non-centered runs.**
