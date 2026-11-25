"""Calibrate RAPM offense + matchup-model defense onto one scale.

Both models are trained on earlier seasons. On a held-out season, every stint
row (one team on offense) gets three terms:

    O   = sum of the 5 scorers' O-RAPM
    DM  = Hungarian-optimal matchup defense of the 5 defenders on the 5 scorers
          (sum of predicted pts/100 per matchup)
    D   = sum of the 5 defenders' D-RAPM (raw: negative = good)

and a weighted least-squares fit (weight = possessions):

    actual pts/100 = base + a*O + b*(DM - dm_mean) + c*D + h*home

a, b near 1 -> the models are on the same scale.
c > 0       -> D-RAPM still explains defense the matchup model misses
               (help defense, rim protection); the data decides how much.

The same fit without DM (RAPM only) is reported for comparison, so we can see
whether the matchup model adds anything beyond RAPM.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.matchups import model as mm
from src.matchups.optimize import Calibration
from src.rapm import build_matrix, fit as rapm_fit


@dataclass
class CalibrationResult:
    cal: Calibration
    home: float
    table: pd.DataFrame          # RAPM-only vs RAPM + matchup, game-level fit

    def to_markdown(self) -> str:
        c = self.cal
        lines = [
            "## Calibration (held-out season)\n",
            "actual pts/100 = base + a·ΣO-RAPM + b·(matchup defense − mean) + c·ΣD-RAPM + h·home\n",
            "| base | a (offense) | b (matchup defense) | c (D-RAPM) | h (home) |",
            "|---|---|---|---|---|",
            f"| {c.base:.2f} | {c.a:.3f} | {c.b:.3f} | {c.c:.3f} | {self.home:+.2f} |",
            "",
            "| Model | Game margin RMSE | Game margin correlation |",
            "|---|---|---|",
        ]
        for r in self.table.itertuples(index=False):
            lines.append(f"| {r.model} | {r.game_rmse:.2f} | {r.game_corr:.3f} |")
        return "\n".join(lines) + "\n"


def _rows(stints: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame({"game_id": stints.game_id, "off": stints.home_players, "def": stints.away_players,
                         "poss": stints.home_poss, "pts": stints.home_pts, "home": 1})
    away = pd.DataFrame({"game_id": stints.game_id, "off": stints.away_players, "def": stints.home_players,
                         "poss": stints.away_poss, "pts": stints.away_pts, "home": 0})
    rows = pd.concat([home, away], ignore_index=True)
    return rows[rows.poss > 0].reset_index(drop=True)


def optimal_matchup_defense(model: mm.MatchupModel, off: np.ndarray, dfn: np.ndarray) -> np.ndarray:
    """off, dfn: (n, 5) player-id arrays. Returns (n,) Hungarian-optimal totals."""
    n = len(off)
    scorers = np.repeat(off, 5, axis=0).reshape(n, 5, 5)        # [r, i, j] -> off[r, j]
    defenders = np.repeat(dfn[:, :, None], 5, axis=2)           # [r, i, j] -> def[r, i]
    M = model.predict_pairs(scorers.ravel(), defenders.ravel()).reshape(n, 5, 5)
    out = np.empty(n)
    for r in range(n):
        i, j = linear_sum_assignment(M[r])
        out[r] = M[r][i, j].sum()
    return out


def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return coef


def _game_fit(rows: pd.DataFrame, pred: np.ndarray) -> tuple[float, float]:
    df = rows.assign(pred_pts=pred * rows.poss / 100)
    g = df.groupby(["game_id", "home"])[["poss", "pts", "pred_pts"]].sum()
    g["actual"], g["pred"] = 100 * g.pts / g.poss, 100 * g.pred_pts / g.poss
    wide = g[["actual", "pred"]].unstack("home").dropna()
    a = wide[("actual", 1)] - wide[("actual", 0)]
    p = wide[("pred", 1)] - wide[("pred", 0)]
    return float(np.sqrt(np.mean((a - p) ** 2))), float(np.corrcoef(a, p)[0, 1])


def run(conn, train_seasons, test_season, decay, rapm_alpha, matchup_alpha,
        kind: str = "poisson", log=print) -> CalibrationResult:
    log(f"Calibration: train {train_seasons} -> test {test_season}")
    weights = build_matrix.season_weights(train_seasons, decay)

    rapm_design = build_matrix.build(build_matrix.load_stints(conn, train_seasons), weights)
    rapm = rapm_fit.fit(rapm_design, rapm_alpha)
    o = pd.Series(rapm.o, index=rapm.player_ids)
    d = pd.Series(rapm.d, index=rapm.player_ids)

    m_rows = mm.load_matchups(conn, train_seasons)
    feats, meta = mm.player_features(conn, m_rows)
    matchup = mm.fit(mm.build(mm.aggregate(m_rows), feats, meta, weights), feats, meta,
                     matchup_alpha, kind)

    rows = _rows(build_matrix.load_stints(conn, [test_season]))
    off = np.array(rows["off"].tolist(), dtype=np.int64)
    dfn = np.array(rows["def"].tolist(), dtype=np.int64)
    log(f"  scoring {len(rows):,} held-out stint rows (Hungarian on each)...")
    O = o.reindex(off.ravel()).fillna(0.0).to_numpy().reshape(off.shape).sum(1)
    D = d.reindex(dfn.ravel()).fillna(0.0).to_numpy().reshape(dfn.shape).sum(1)
    DM = optimal_matchup_defense(matchup, off, dfn)

    poss = rows.poss.to_numpy(float)
    y = 100.0 * rows.pts.to_numpy(float) / poss
    home = rows.home.to_numpy(float)
    dm_mean = float(np.sum(poss * DM) / poss.sum())
    ones = np.ones(len(y))

    full = _wls(np.column_stack([ones, O, DM - dm_mean, D, home]), y, poss)
    base_only = _wls(np.column_stack([ones, O, D, home]), y, poss)

    pred_full = np.column_stack([ones, O, DM - dm_mean, D, home]) @ full
    pred_rapm = np.column_stack([ones, O, D, home]) @ base_only
    table = pd.DataFrame([
        {"model": "RAPM only", **dict(zip(["game_rmse", "game_corr"], _game_fit(rows, pred_rapm)))},
        {"model": "RAPM + matchup defense", **dict(zip(["game_rmse", "game_corr"], _game_fit(rows, pred_full)))},
    ])
    cal = Calibration(base=float(full[0]), a=float(full[1]), b=float(full[2]),
                      c=float(full[3]), dm_mean=dm_mean)
    return CalibrationResult(cal=cal, home=float(full[4]), table=table)
