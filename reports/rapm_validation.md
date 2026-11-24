# RAPM out-of-sample validation

- Trained on: 2023-24, 2024-25 (recency-weighted)
- Tested on: 2025-26 (never seen by the model)
- Ridge penalty (alpha, chosen by game-level CV): 3,162

| Predictor | Stint RMSE (pts/100) | Game margin RMSE (pts/100) | Game margin correlation |
|---|---|---|---|
| Baseline (league avg) | 66.30 | 16.66 | -0.029 |
| Raw +/- | 66.24 | 15.99 | 0.285 |
| RAPM | 66.19 | 15.61 | 0.341 |

Game-margin error: RAPM is 2.4% lower than raw +/- and 6.3% lower than the baseline.
