"""Out-of-sample test for the matchup model.

Train on earlier seasons, predict a season the model never saw, and compare:

    League average                 every matchup = league scoring rate
    Linear (additive), full        scorer + defender + size/style, added
    Poisson: scorer only           multiplicative, scorer skill only
    Poisson: scorer + defender     + defender skill
    Poisson: full                  + size/style terms
    Poisson: full + team context   + team-season context terms during fitting
                                   (the test season's teams are new, so context
                                   is not used when predicting it: this checks
                                   whether de-confounded player skills carry over)

Scored on every held-out row (weighted RMSE in pts/100 and Poisson deviance,
both very noisy) and on pairs with enough held-out possessions to judge
(correlation of predicted vs actual pts/100).
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
    alphas: dict
    feat_coef: np.ndarray       # Poisson full model, log-scale per feature unit
    table: pd.DataFrame

    def to_markdown(self) -> str:
        pct = ", ".join(f"{n} {100 * (np.exp(w) - 1):+.2f}% per unit"
                        for n, w in zip(mm.FEATURES, self.feat_coef))
        lines = [
            "# Matchup model out-of-sample validation\n",
            f"- Trained on: {', '.join(self.train_seasons)} (recency-weighted)",
            f"- Tested on: {self.test_season} (never seen by the model)",
            f"- Ridge penalties (chosen by game-level CV): linear {self.alphas['linear']:,.3g}, "
            f"Poisson {self.alphas['poisson']:.3g}, Poisson + context {self.alphas['poisson+context']:.3g}",
            f"- Poisson size/style effects: {pct}\n",
            f"| Model | Row RMSE (pts/100) | Row Poisson deviance | Pair correlation ({MIN_PAIR_POSS}+ poss) | Pairs |",
            "|---|---|---|---|---|",
        ]
        for r in self.table.itertuples(index=False):
            lines.append(f"| {r.model} | {r.rmse:.2f} | {r.deviance:.4f} | {r.pair_corr:.3f} | {r.n_pairs:,} |")
        return "\n".join(lines) + "\n"


def _pair_corr(rows: pd.DataFrame, pred: np.ndarray) -> tuple[float, int]:
    df = pd.DataFrame({"o": rows["off_player_id"].to_numpy(), "d": rows["def_player_id"].to_numpy(),
                       "poss": rows["partial_poss"].to_numpy(float), "pts": rows["player_pts"].to_numpy(float)})
    df["pred_pts"] = pred * df["poss"] / 100
    g = df.groupby(["o", "d"])[["poss", "pts", "pred_pts"]].sum()
    g = g[g["poss"] >= MIN_PAIR_POSS]
    actual, predicted = 100 * g["pts"] / g["poss"], 100 * g["pred_pts"] / g["poss"]
    corr = float(np.corrcoef(actual, predicted)[0, 1]) if predicted.std() > 0 else float("nan")
    return corr, int(len(g))


def run(conn, train_seasons, test_season, decay=0.5, n_splits=5, log=print) -> MatchupValidation:
    log(f"Matchup validation: train {train_seasons} -> test {test_season}")
    weights = season_weights(train_seasons, decay)
    train_rows = mm.load_matchups(conn, train_seasons)
    test_rows = mm.load_matchups(conn, [test_season])
    feats, meta = mm.player_features(conn, train_rows)
    design = mm.build(mm.aggregate(train_rows), feats, meta, weights)
    log(f"  train rows {len(train_rows):,} ({len(design.rate):,} after aggregating), "
        f"test rows {len(test_rows):,}")

    alphas = {
        "linear": mm.cross_validate(train_rows, feats, meta, weights, "linear", n_splits=n_splits,
                                    which="full", log=log)[0],
        "poisson": mm.cross_validate(train_rows, feats, meta, weights, "poisson", n_splits=n_splits,
                                     which="full", log=log)[0],
        "poisson+context": mm.cross_validate(train_rows, feats, meta, weights, "poisson", n_splits=n_splits,
                                             which="full+context", log=log)[0],
    }

    models = {
        "Linear (additive), full": mm.fit(design, feats, meta, alphas["linear"], "linear", "full"),
        "Poisson: scorer only": mm.fit(design, feats, meta, alphas["poisson"], "poisson", "scorer"),
        "Poisson: scorer + defender": mm.fit(design, feats, meta, alphas["poisson"], "poisson", "scorer+defender"),
        "Poisson: full (+ size/style)": mm.fit(design, feats, meta, alphas["poisson"], "poisson", "full"),
        "Poisson: full + team context": mm.fit(design, feats, meta, alphas["poisson+context"], "poisson",
                                               "full+context"),
    }

    poss = test_rows["partial_poss"].to_numpy(float)
    pts = test_rows["player_pts"].to_numpy(float)
    league = 100 * float(np.sum(design.rate * design.w) / np.sum(design.w))
    preds = {"League average": np.full(len(test_rows), league)}
    preds.update({name: m.predict_rows(test_rows) for name, m in models.items()})

    rows = []
    for name, pred in preds.items():
        corr, n_pairs = _pair_corr(test_rows, pred)
        rows.append({"model": name,
                     "rmse": mm.weighted_rmse(100 * pts / poss, pred, poss),
                     "deviance": mm.poisson_deviance(pts, poss, pred),
                     "pair_corr": corr, "n_pairs": n_pairs})
    return MatchupValidation(train_seasons, test_season, alphas,
                             models["Poisson: full + team context"].feat_coef, pd.DataFrame(rows))
