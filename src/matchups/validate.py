"""Out-of-sample test for the matchup model.

Train on earlier seasons, predict a season the model never saw, and compare:
    league average           every matchup = intercept
    scorer only              + scorer skill
    scorer + defender        + defender skill
    full                     + size/style pair features

Scored on every test row (weighted RMSE, very noisy) and on pairs with enough
test possessions to judge (correlation of predicted vs actual pts/100).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.matchups import model as mm
from src.rapm.build_matrix import season_weights

MIN_PAIR_POSS = 25


@dataclass
class MatchupValidation:
    train_seasons: list[str]
    test_season: str
    alpha: float
    feat_coef: np.ndarray
    table: pd.DataFrame

    def to_markdown(self) -> str:
        lines = [
            "# Matchup model out-of-sample validation\n",
            f"- Trained on: {', '.join(self.train_seasons)} (recency-weighted)",
            f"- Tested on: {self.test_season} (never seen by the model)",
            f"- Ridge penalty (chosen by game-level CV): {self.alpha:,.0f}",
            "- Size/style weights (pts per 100 matchup possessions): "
            + ", ".join(f"{n} {w:+.3f}" for n, w in zip(mm.FEATURES, self.feat_coef)) + "\n",
            f"| Model | Row RMSE (pts/100) | Pair correlation (pairs with {MIN_PAIR_POSS}+ test poss) | Pairs |",
            "|---|---|---|---|",
        ]
        for r in self.table.itertuples(index=False):
            lines.append(f"| {r.model} | {r.row_rmse:.2f} | {r.pair_corr:.3f} | {r.n_pairs:,} |")
        return "\n".join(lines) + "\n"


def run(conn, train_seasons, test_season, decay=0.5, alphas=mm.DEFAULT_ALPHAS,
        n_splits=5, log=print) -> MatchupValidation:
    log(f"Matchup validation: train {train_seasons} -> test {test_season}")
    train_rows = mm.load_matchups(conn, train_seasons)
    test_rows = mm.load_matchups(conn, [test_season])
    feats, meta = mm.player_features(conn, train_rows)
    train = mm.build(train_rows, feats, meta, season_weights(train_seasons, decay))
    log(f"  train rows {len(train.y):,}, test rows {len(test_rows):,}")
    alpha, _ = mm.cross_validate(train, alphas, n_splits, log=log)

    off = test_rows["off_player_id"].to_numpy()
    dfn = test_rows["def_player_id"].to_numpy()
    poss = test_rows["partial_poss"].to_numpy(dtype=float)
    y = 100.0 * test_rows["player_pts"].to_numpy(dtype=float) / poss

    preds, full = {}, None
    for which, label in [("scorer", "Scorer only"), ("scorer+defender", "Scorer + defender"),
                         ("full", "Full (+ size/style)")]:
        m = mm.fit(train, feats, meta, alpha, which)
        preds[label] = m.predict_pairs(off, dfn)
        if which == "full":
            full = m
            preds = {"League average": np.full(len(y), m.intercept), **preds}

    pair_key = pd.Series(list(zip(off, dfn)))
    rows = []
    for label, pred in preds.items():
        df = pd.DataFrame({"pair": pair_key, "poss": poss, "pts": y * poss / 100,
                           "pred_pts": pred * poss / 100})
        g = df.groupby("pair")[["poss", "pts", "pred_pts"]].sum()
        g = g[g["poss"] >= MIN_PAIR_POSS]
        actual, predicted = 100 * g["pts"] / g["poss"], 100 * g["pred_pts"] / g["poss"]
        corr = float(np.corrcoef(actual, predicted)[0, 1]) if predicted.std() > 0 else float("nan")
        rows.append({"model": label, "row_rmse": mm.weighted_rmse(y, pred, poss),
                     "pair_corr": corr, "n_pairs": int(len(g))})
    return MatchupValidation(train_seasons, test_season, alpha, full.feat_coef, pd.DataFrame(rows))
