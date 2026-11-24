"""Matchup model: how many points does scorer S produce per 100 possessions
while guarded by defender D?

    y = 100 * scorer_points / matchup_possessions        (one row per game x pair)

    y ~ intercept + scorer_skill[S] + defender_skill[D]
                  + w1 * height_diff            (defender - scorer, inches)
                  + w2 * weight_diff / 10       (defender - scorer, lbs / 10)
                  + w3 * height_diff^2 / 10     (size mismatch in either direction)
                  + w4 * three_rate_c * def_height_c
                         (scorer's 3PA share, centered) x (defender height, centered)

Note: height_diff and weight_diff are additive (defender part + scorer part),
so for players with lots of data the player skills absorb them; they mainly
act as size-based priors for players with little data. The mismatch and
3PA x defender-height terms are true pair interactions.

Fit as a weighted ridge regression (weight = matchup possessions x season
recency weight), penalty chosen by cross-validation split by game. Matchup
possessions count every possession the pair was matched up, not only the ones
the scorer used, so a scorer's shot volume is already inside y.

Every number needed to predict a pair is stored (player skills + features +
the few scalars below), so the dashboard can rebuild any prediction.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

DEFAULT_ALPHAS = np.logspace(1.5, 4.5, 9)    # ~32 ... ~31,600
FEATURE_SCALE = 10.0      # lightly penalize the pair features (penalty / 100)
FEATURES = ["height_diff", "weight_diff_10", "mismatch_sq_10", "three_x_def_height"]
THREE_PRIOR_FGA = 50      # shrink each scorer's 3PA share toward league with 50 FGA


# ---------------------------------------------------------------- data ----

def load_matchups(conn, seasons: list[str]) -> pd.DataFrame:
    sql = """
        SELECT m.game_id, g.season, m.off_player_id, m.def_player_id,
               m.partial_poss, m.player_pts, m.fga, m.fg3a
        FROM matchups m JOIN games g USING (game_id)
        WHERE g.season = ANY(%s) AND m.partial_poss > 0
    """
    with conn.cursor() as cur:
        cur.execute(sql, (list(seasons),))
        cols = [c[0] for c in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
    for c in ("partial_poss", "player_pts", "fga", "fg3a"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def player_features(conn, matchups: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Height, weight and (shrunk) 3PA share for every player in the data."""
    with conn.cursor() as cur:
        cur.execute("SELECT player_id, height_inches, weight_lbs FROM players")
        bio = pd.DataFrame(cur.fetchall(), columns=["player_id", "height_inches", "weight_lbs"])
    bio = bio.set_index("player_id").astype(float)

    shots = matchups.groupby("off_player_id")[["fga", "fg3a"]].sum()
    league_rate = float(shots["fg3a"].sum() / max(shots["fga"].sum(), 1.0))
    rate = (shots["fg3a"] + league_rate * THREE_PRIOR_FGA) / (shots["fga"] + THREE_PRIOR_FGA)

    ids = np.union1d(matchups["off_player_id"].unique(), matchups["def_player_id"].unique())
    feats = pd.DataFrame(index=pd.Index(ids, name="player_id"))
    feats["height_inches"] = bio["height_inches"].reindex(ids)
    feats["weight_lbs"] = bio["weight_lbs"].reindex(ids)
    feats["three_rate"] = rate.reindex(ids)

    meta = {
        "height_mean": float(feats["height_inches"].mean()),
        "weight_mean": float(feats["weight_lbs"].mean()),
        "three_rate_mean": league_rate,
    }
    feats["height_inches"] = feats["height_inches"].fillna(meta["height_mean"])
    feats["weight_lbs"] = feats["weight_lbs"].fillna(meta["weight_mean"])
    feats["three_rate"] = feats["three_rate"].fillna(league_rate)
    return feats, meta


def pair_features(off_ids, def_ids, feats: pd.DataFrame, meta: dict) -> np.ndarray:
    """(n, 4) array of the pair features. Unknown players get league means."""
    def col(ids, name, default):
        return feats[name].reindex(ids).fillna(default).to_numpy(dtype=float)

    h_o = col(off_ids, "height_inches", meta["height_mean"])
    h_d = col(def_ids, "height_inches", meta["height_mean"])
    w_o = col(off_ids, "weight_lbs", meta["weight_mean"])
    w_d = col(def_ids, "weight_lbs", meta["weight_mean"])
    t_o = col(off_ids, "three_rate", meta["three_rate_mean"])
    return np.column_stack([
        h_d - h_o,
        (w_d - w_o) / 10.0,
        (h_d - h_o) ** 2 / 10.0,
        (t_o - meta["three_rate_mean"]) * (h_d - meta["height_mean"]),
    ])


@dataclass
class MatchupDesign:
    X: sp.csr_matrix          # [scorer P | defender P | features]
    y: np.ndarray
    w: np.ndarray
    poss: np.ndarray
    groups: np.ndarray
    player_ids: np.ndarray

    @property
    def n_players(self) -> int:
        return len(self.player_ids)


def build(matchups: pd.DataFrame, feats: pd.DataFrame, meta: dict,
          weights: dict[str, float] | None = None,
          player_ids: np.ndarray | None = None) -> MatchupDesign:
    off = matchups["off_player_id"].to_numpy(dtype=np.int64)
    dfn = matchups["def_player_id"].to_numpy(dtype=np.int64)
    if player_ids is None:
        player_ids = np.union1d(off, dfn)
    player_ids = np.asarray(player_ids, dtype=np.int64)
    P, n = len(player_ids), len(matchups)

    def idx(ids):
        pos = np.clip(np.searchsorted(player_ids, ids), 0, P - 1)
        return pos, player_ids[pos] == ids

    o_pos, o_ok = idx(off)
    d_pos, d_ok = idx(dfn)
    F = pair_features(off, dfn, feats, meta) * FEATURE_SCALE
    rows = np.arange(n)
    k = len(FEATURES)
    r = np.concatenate([rows[o_ok], rows[d_ok], np.repeat(rows, k)])
    c = np.concatenate([o_pos[o_ok], P + d_pos[d_ok], np.tile(2 * P + np.arange(k), n)])
    v = np.concatenate([np.ones(o_ok.sum()), np.ones(d_ok.sum()), F.ravel()])
    X = sp.csr_matrix((v, (r, c)), shape=(n, 2 * P + k))

    poss = matchups["partial_poss"].to_numpy(dtype=float)
    y = 100.0 * matchups["player_pts"].to_numpy(dtype=float) / poss
    sw = (matchups["season"].map(weights).fillna(0.0).to_numpy(dtype=float)
          if weights else np.ones(n))
    return MatchupDesign(X=X, y=y, w=poss * sw, poss=poss,
                         groups=matchups["game_id"].to_numpy(), player_ids=player_ids)


# --------------------------------------------------------------- model ----

@dataclass
class MatchupModel:
    alpha: float
    intercept: float
    player_ids: np.ndarray
    scorer: np.ndarray
    defender: np.ndarray
    feat_coef: np.ndarray                 # per unit of the (unscaled) features
    feats: pd.DataFrame
    meta: dict
    poss_off: np.ndarray = field(default=None)
    poss_def: np.ndarray = field(default=None)

    def _skill(self, ids, arr) -> np.ndarray:
        s = pd.Series(arr, index=self.player_ids)
        return s.reindex(np.asarray(ids)).fillna(0.0).to_numpy()

    def predict_pairs(self, off_ids, def_ids) -> np.ndarray:
        """Points per 100 matchup possessions for each (scorer, defender) pair."""
        off_ids, def_ids = np.asarray(off_ids), np.asarray(def_ids)
        F = pair_features(off_ids, def_ids, self.feats, self.meta)
        return (self.intercept + self._skill(off_ids, self.scorer)
                + self._skill(def_ids, self.defender) + F @ self.feat_coef)

    def matrix(self, defenders, scorers) -> np.ndarray:
        """M[i, j] = predicted pts/100 if defenders[i] guards scorers[j]."""
        d = np.repeat(np.asarray(defenders), len(scorers))
        s = np.tile(np.asarray(scorers), len(defenders))
        return self.predict_pairs(s, d).reshape(len(defenders), len(scorers))

    def params(self) -> dict:
        p = {"intercept": self.intercept, "alpha": self.alpha}
        p.update({f"w_{n}": float(w) for n, w in zip(FEATURES, self.feat_coef)})
        p.update(self.meta)
        return p


def _ridge(alpha):
    return Ridge(alpha=alpha, fit_intercept=True, solver="sparse_cg", tol=1e-6, max_iter=5000)


def _cols(design: MatchupDesign, which: str):
    P = design.n_players
    if which == "scorer":
        return np.arange(P)
    if which == "scorer+defender":
        return np.arange(2 * P)
    return np.arange(2 * P + len(FEATURES))


def weighted_rmse(y, pred, w) -> float:
    return float(np.sqrt(np.sum(w * (y - pred) ** 2) / np.sum(w)))


def cross_validate(design: MatchupDesign, alphas=DEFAULT_ALPHAS, n_splits=5, log=print):
    folds = list(GroupKFold(n_splits=n_splits).split(design.X, design.y, design.groups))
    out = []
    for a in alphas:
        errs = []
        for tr, te in folds:
            m = _ridge(a).fit(design.X[tr], design.y[tr], sample_weight=design.w[tr])
            errs.append(weighted_rmse(design.y[te], m.predict(design.X[te]), design.w[te]))
        out.append({"alpha": float(a), "cv_rmse": float(np.mean(errs))})
        log(f"  alpha={a:>9.1f}  cv_rmse={np.mean(errs):.4f}")
    table = pd.DataFrame(out)
    best = table.loc[table["cv_rmse"].idxmin()]
    if best["alpha"] in (table["alpha"].min(), table["alpha"].max()):
        log("  WARNING: best alpha is at the edge of the grid; consider widening it.")
    return float(best["alpha"]), table


def fit(design: MatchupDesign, feats, meta, alpha: float, which: str = "full") -> MatchupModel:
    cols = _cols(design, which)
    m = _ridge(alpha).fit(design.X[:, cols], design.y, sample_weight=design.w)
    coef = np.zeros(design.X.shape[1])
    coef[cols] = m.coef_
    P = design.n_players
    poss_by_col = design.X[:, :2 * P].T @ design.poss
    return MatchupModel(
        alpha=alpha, intercept=float(m.intercept_), player_ids=design.player_ids,
        scorer=coef[:P], defender=coef[P:2 * P],
        feat_coef=coef[2 * P:] * FEATURE_SCALE, feats=feats, meta=meta,
        poss_off=poss_by_col[:P], poss_def=poss_by_col[P:])
