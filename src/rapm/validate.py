"""Out-of-sample test: train on earlier seasons, predict a season the model
never saw, and compare against two simpler predictors.

    baseline   every possession scores the league average
    raw +/-    league average + mean of the 5 offensive players' raw on-court
               offensive rating (vs average) + mean of the 5 defenders' raw
               on-court defensive rating (vs average). No adjustment for
               teammates/opponents, no shrinkage.
    RAPM       the ridge model

Scored two ways:
    stint level   weighted RMSE of pts per 100 on every test stint-row
    game level    given who actually played, how well does each predictor
                  explain each game's final margin (pts per 100)? Stints are
                  very noisy; whole games show the signal more clearly.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.rapm import build_matrix, fit as rapm_fit


@dataclass
class ValidationResult:
    train_seasons: list[str]
    test_season: str
    alpha: float
    table: pd.DataFrame      # one row per predictor

    def to_markdown(self) -> str:
        lines = [
            f"# RAPM out-of-sample validation\n",
            f"- Trained on: {', '.join(self.train_seasons)} (recency-weighted)",
            f"- Tested on: {self.test_season} (never seen by the model)",
            f"- Ridge penalty (alpha, chosen by game-level CV): {self.alpha:,.0f}\n",
            "| Predictor | Stint RMSE (pts/100) | Game margin RMSE (pts/100) | Game margin correlation |",
            "|---|---|---|---|",
        ]
        for r in self.table.itertuples(index=False):
            lines.append(f"| {r.predictor} | {r.stint_rmse:.2f} | {r.game_rmse:.2f} | {r.game_corr:.3f} |")
        rapm = self.table.set_index("predictor").loc["RAPM"]
        raw = self.table.set_index("predictor").loc["Raw +/-"]
        base = self.table.set_index("predictor").loc["Baseline (league avg)"]
        lines += [
            "",
            f"Game-margin error: RAPM is {100 * (1 - rapm.game_rmse / raw.game_rmse):.1f}% lower "
            f"than raw +/- and {100 * (1 - rapm.game_rmse / base.game_rmse):.1f}% lower than the baseline.",
        ]
        return "\n".join(lines) + "\n"


def _raw_plus_minus(train: build_matrix.Design, avg: float) -> tuple[np.ndarray, np.ndarray]:
    """Each player's raw on-court offensive and defensive rating minus league avg."""
    P = train.n_players
    pts = train.y * train.poss / 100.0
    pts_by_col = train.X.T @ (pts * train.w / train.poss)
    poss_by_col = train.X.T @ train.w
    with np.errstate(divide="ignore", invalid="ignore"):
        rtg = np.where(poss_by_col > 0, 100.0 * pts_by_col / poss_by_col, avg)
    return rtg[:P] - avg, rtg[P:2 * P] - avg


def _game_margins(design: build_matrix.Design, pred: np.ndarray, home_flag: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"game_id": design.groups, "home": home_flag, "poss": design.poss,
                       "pts": design.y * design.poss / 100.0, "pred_pts": pred * design.poss / 100.0})
    g = df.groupby(["game_id", "home"])[["poss", "pts", "pred_pts"]].sum()
    g["actual"] = 100 * g["pts"] / g["poss"]
    g["pred"] = 100 * g["pred_pts"] / g["poss"]
    wide = g[["actual", "pred"]].unstack("home")
    return pd.DataFrame({
        "actual_margin": wide[("actual", 1)] - wide[("actual", 0)],
        "pred_margin": wide[("pred", 1)] - wide[("pred", 0)],
    }).dropna()


def run(conn, train_seasons: list[str], test_season: str, decay: float = 0.5,
        alphas=rapm_fit.DEFAULT_ALPHAS, n_splits: int = 5, log=print) -> ValidationResult:
    log(f"Validation: train {train_seasons} -> test {test_season}")
    train_stints = build_matrix.load_stints(conn, train_seasons)
    test_stints = build_matrix.load_stints(conn, [test_season])
    train = build_matrix.build(train_stints, build_matrix.season_weights(train_seasons, decay))
    test = build_matrix.build(test_stints, None, player_ids=train.player_ids)
    log(f"  train rows {len(train.y):,}, test rows {len(test.y):,}")

    alpha, _ = rapm_fit.cross_validate(train, alphas, n_splits, log=log)
    model = rapm_fit.fit(train, alpha)

    avg = float(np.sum(train.w * train.y) / np.sum(train.w))
    o_raw, d_raw = _raw_plus_minus(train, avg)
    P = train.n_players
    preds = {
        "Baseline (league avg)": np.full(len(test.y), avg),
        "Raw +/-": avg + (test.X[:, :P] @ o_raw) / 5.0 + (test.X[:, P:2 * P] @ d_raw) / 5.0,
        "RAPM": model.predict(test),
    }
    home_flag = np.asarray(test.X[:, test.home_col].todense()).ravel().astype(int)

    rows = []
    for name, pred in preds.items():
        games = _game_margins(test, pred, home_flag)
        rows.append({
            "predictor": name,
            "stint_rmse": rapm_fit.weighted_rmse(test.y, pred, test.poss),
            "game_rmse": float(np.sqrt(np.mean((games.actual_margin - games.pred_margin) ** 2))),
            "game_corr": float(np.corrcoef(games.actual_margin, games.pred_margin)[0, 1])
                         if games.pred_margin.std() > 0 else float("nan"),
        })
    return ValidationResult(train_seasons, test_season, alpha, pd.DataFrame(rows))
