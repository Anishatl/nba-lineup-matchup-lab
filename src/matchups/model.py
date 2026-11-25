"""Matchup model: how many points does scorer S produce per 100 possessions
while guarded by defender D?

Two model shapes, both ridge-regularized and cross-validated by game:

  linear  (additive)
      pts/100 = intercept + scorer[S] + defender[D] + size/style terms
      A defender removes the same number of points from anyone he guards, so
      who-guards-whom only matters through the size/style terms.

  poisson (multiplicative, default)
      log(pts per poss) = intercept + scorer[S] + defender[D] + size/style terms
      pts/100 = 100 * e^intercept * e^scorer * e^defender * e^size
      A defender removes a *share* of his man's scoring (e.g. x0.88 = -12%),
      so he matters more against high-volume scorers and the assignment
      optimizer has something real to optimize.

Size/style terms (true pair interactions, centered on the average matchup so
they are ~0 for a typical pair):
      mismatch_sq_10       (defender height - scorer height)^2 / 10
      three_x_def_height   (scorer 3PA share - league) x (defender height - league)
Additive height/weight differences are deliberately left out: they split into
"defender part + scorer part", so they compete with the player skills and make
the skills hard to read.

Team context (fit only): each row also gets a term for the defending
team-season, penalized like the player terms. Points against a defender depend
a lot on his teammates (help defense, rim protection, scheme); without this term
that team effect gets credited to individual defenders. (An offensive
team-season term was tested and dropped: a team's offense is largely its best
scorer, so the term took credit away from individual scorers.) Predictions and the optimizer use the
player terms only: in a multiplicative model a team term scales every matchup
for that team equally, so it can't change which lineup or assignment is best.

Rows are aggregated to (scorer, defender, season, teams) before fitting. With no
game-level inputs this gives exactly the same fit, much faster. Cross-validation
still splits raw rows by game.

Matchup possessions count every possession the pair was matched up, not only
the ones the scorer used, so a scorer's shot volume is already inside the target.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

FEATURES = ["mismatch_sq_10", "three_x_def_height"]
CTX_COLUMNS = ("off_ctx", "def_ctx")   # whichever are present get context terms (default: def_ctx only)
FEATURE_SCALE = 10.0       # lightly penalize the pair features (penalty / 100)
THREE_PRIOR_FGA = 50       # shrink each scorer's 3PA share toward league with 50 FGA
ALPHAS = {
    "linear": np.logspace(1.5, 4.5, 9),     # ~32 ... ~31,600
    "poisson": np.logspace(-6.5, -2.5, 9),  # sklearn-style alpha (loss scaled by 1/sum(weights))
}


# ---------------------------------------------------------------- data ----

def load_matchups(conn, seasons: list[str]) -> pd.DataFrame:
    # Offensive team = whichever team in this game the scorer played for that season.
    sql = """
        SELECT m.game_id, g.season, m.off_player_id, m.def_player_id,
               m.partial_poss, m.player_pts, m.fga, m.fg3a,
               CASE WHEN EXISTS (SELECT 1 FROM player_team_season p
                                 WHERE p.player_id = m.off_player_id AND p.season = g.season
                                   AND p.team_id = g.home_team_id)
                    THEN g.home_team_id ELSE g.away_team_id END AS off_team,
               CASE WHEN EXISTS (SELECT 1 FROM player_team_season p
                                 WHERE p.player_id = m.off_player_id AND p.season = g.season
                                   AND p.team_id = g.home_team_id)
                    THEN g.away_team_id ELSE g.home_team_id END AS def_team
        FROM matchups m JOIN games g USING (game_id)
        WHERE g.season = ANY(%s) AND m.partial_poss > 0
    """
    with conn.cursor() as cur:
        cur.execute(sql, (list(seasons),))
        cols = [c[0] for c in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
    for c in ("partial_poss", "player_pts", "fga", "fg3a"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["off_player_id"] = df["off_player_id"].astype(np.int64)
    df["def_player_id"] = df["def_player_id"].astype(np.int64)
    df["def_ctx"] = "d|" + df["def_team"].astype(str) + "|" + df["season"]
    return df


def player_features(conn, matchups: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Height, weight and (shrunk) 3PA share for every player, plus the
    constants needed to compute and center the pair features."""
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

    # Center each feature on the possession-weighted average matchup
    raw = _raw_pair_features(matchups["off_player_id"].to_numpy(),
                             matchups["def_player_id"].to_numpy(), feats, meta)
    w = matchups["partial_poss"].to_numpy(dtype=float)
    for k, name in enumerate(FEATURES):
        meta[f"center_{name}"] = float(np.sum(w * raw[:, k]) / max(w.sum(), 1e-9))
    return feats, meta


def _raw_pair_features(off_ids, def_ids, feats: pd.DataFrame, meta: dict) -> np.ndarray:
    def col(ids, name, default):
        return feats[name].reindex(np.asarray(ids)).fillna(default).to_numpy(dtype=float)

    h_o = col(off_ids, "height_inches", meta["height_mean"])
    h_d = col(def_ids, "height_inches", meta["height_mean"])
    t_o = col(off_ids, "three_rate", meta["three_rate_mean"])
    return np.column_stack([
        (h_d - h_o) ** 2 / 10.0,
        (t_o - meta["three_rate_mean"]) * (h_d - meta["height_mean"]),
    ])


def pair_features(off_ids, def_ids, feats: pd.DataFrame, meta: dict) -> np.ndarray:
    """(n, len(FEATURES)) centered pair features. Unknown players get league means."""
    raw = _raw_pair_features(off_ids, def_ids, feats, meta)
    centers = np.array([meta.get(f"center_{n}", 0.0) for n in FEATURES])
    return raw - centers


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    """Sum possessions and points per (scorer, defender, season, teams)."""
    keys = ["off_player_id", "def_player_id", "season"]
    keys += [k for k in CTX_COLUMNS if k in rows.columns]
    return rows.groupby(keys, as_index=False)[["partial_poss", "player_pts"]].sum()


@dataclass
class MatchupDesign:
    X: sp.csr_matrix          # [scorer P | defender P | features | team contexts]
    rate: np.ndarray          # points per possession
    w: np.ndarray             # possessions x season recency weight
    poss: np.ndarray
    player_ids: np.ndarray
    ctx_keys: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))

    @property
    def n_players(self) -> int:
        return len(self.player_ids)


def build(rows: pd.DataFrame, feats: pd.DataFrame, meta: dict,
          weights: dict[str, float] | None = None,
          player_ids: np.ndarray | None = None,
          ctx_keys: np.ndarray | None = None) -> MatchupDesign:
    """Build the design matrix from (already aggregated or raw) rows.
    Team-context columns are added when the rows carry off_ctx/def_ctx."""
    off = rows["off_player_id"].to_numpy(dtype=np.int64)
    dfn = rows["def_player_id"].to_numpy(dtype=np.int64)
    if player_ids is None:
        player_ids = np.union1d(off, dfn)
    player_ids = np.asarray(player_ids, dtype=np.int64)
    P, n, k = len(player_ids), len(rows), len(FEATURES)

    def idx(ids):
        pos = np.clip(np.searchsorted(player_ids, ids), 0, P - 1)
        return pos, player_ids[pos] == ids

    o_pos, o_ok = idx(off)
    d_pos, d_ok = idx(dfn)
    F = pair_features(off, dfn, feats, meta) * FEATURE_SCALE
    r_idx = np.arange(n)
    r = np.concatenate([r_idx[o_ok], r_idx[d_ok], np.repeat(r_idx, k)])
    c = np.concatenate([o_pos[o_ok], P + d_pos[d_ok], np.tile(2 * P + np.arange(k), n)])
    v = np.concatenate([np.ones(o_ok.sum()), np.ones(d_ok.sum()), F.ravel()])
    X = sp.csr_matrix((v, (r, c)), shape=(n, 2 * P + k))

    ctx_cols = [c_ for c_ in CTX_COLUMNS if c_ in rows.columns]
    if ctx_cols:
        if ctx_keys is None:
            ctx_keys = np.unique(np.concatenate([rows[c_].unique() for c_ in ctx_cols]))
        ctx_keys = np.asarray(ctx_keys, dtype=object)
        lookup = pd.Series(np.arange(len(ctx_keys)), index=ctx_keys)
        rr, cc = [], []
        for c_ in ctx_cols:
            pos = lookup.reindex(rows[c_]).to_numpy()
            ok = ~np.isnan(pos)
            rr.append(r_idx[ok])
            cc.append(pos[ok].astype(int))
        rr, cc = np.concatenate(rr), np.concatenate(cc)
        C = sp.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(n, len(ctx_keys)))
        X = sp.hstack([X, C], format="csr")
    else:
        ctx_keys = np.array([], dtype=object)

    poss = rows["partial_poss"].to_numpy(dtype=float)
    rate = rows["player_pts"].to_numpy(dtype=float) / poss
    sw = (rows["season"].map(weights).fillna(0.0).to_numpy(dtype=float) if weights else np.ones(n))
    return MatchupDesign(X=X, rate=rate, w=poss * sw, poss=poss, player_ids=player_ids,
                         ctx_keys=ctx_keys)


# --------------------------------------------------------------- model ----

@dataclass
class MatchupModel:
    kind: str                             # 'linear' or 'poisson'
    alpha: float
    intercept: float
    player_ids: np.ndarray
    scorer: np.ndarray
    defender: np.ndarray
    feat_coef: np.ndarray                 # per unit of the (unscaled, centered) features
    feats: pd.DataFrame
    meta: dict
    poss_off: np.ndarray = field(default=None)
    poss_def: np.ndarray = field(default=None)
    size_strength: float = 1.0            # scales the size/style terms (1 = as fitted, 0 = off)
    ctx_keys: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    ctx_coef: np.ndarray = field(default_factory=lambda: np.array([]))

    def _skill(self, ids, arr) -> np.ndarray:
        s = pd.Series(arr, index=self.player_ids)
        return s.reindex(np.asarray(ids)).fillna(0.0).to_numpy()

    def predict_pairs(self, off_ids, def_ids) -> np.ndarray:
        """Predicted points per 100 matchup possessions for each (scorer, defender)."""
        off_ids, def_ids = np.asarray(off_ids), np.asarray(def_ids)
        eta = (self.intercept + self._skill(off_ids, self.scorer)
               + self._skill(def_ids, self.defender)
               + pair_features(off_ids, def_ids, self.feats, self.meta)
               @ (self.feat_coef * self.size_strength))
        return 100.0 * np.exp(eta) if self.kind == "poisson" else eta

    def predict_rows(self, rows: pd.DataFrame) -> np.ndarray:
        """Like predict_pairs, but adds the team-context terms for rows whose
        team-seasons the model has seen (used for scoring held-out games)."""
        off_ids = rows["off_player_id"].to_numpy()
        def_ids = rows["def_player_id"].to_numpy()
        ctx_cols = [c for c in CTX_COLUMNS if c in rows.columns]
        if not len(self.ctx_keys) or not ctx_cols:
            return self.predict_pairs(off_ids, def_ids)
        ctx = pd.Series(self.ctx_coef, index=self.ctx_keys)
        extra = sum(ctx.reindex(rows[c]).fillna(0.0).to_numpy() for c in ctx_cols)
        eta = (self.intercept + self._skill(off_ids, self.scorer) + self._skill(def_ids, self.defender)
               + pair_features(off_ids, def_ids, self.feats, self.meta) @ (self.feat_coef * self.size_strength)
               + extra)
        return 100.0 * np.exp(eta) if self.kind == "poisson" else eta

    def team_context(self) -> pd.DataFrame:
        """Fitted team-season context effects as % (Poisson) or pts/100 (linear)."""
        if not len(self.ctx_keys):
            return pd.DataFrame(columns=["side", "team_id", "season", "effect"])
        parts = pd.Series(self.ctx_keys).str.split("|", expand=True)
        eff = 100 * (np.exp(self.ctx_coef) - 1) if self.kind == "poisson" else self.ctx_coef
        return pd.DataFrame({"side": parts[0].map({"o": "offense", "d": "defense"}),
                             "team_id": parts[1].astype(int), "season": parts[2], "effect": eff})

    def matrix(self, defenders, scorers) -> np.ndarray:
        """M[i, j] = predicted pts/100 if defenders[i] guards scorers[j]."""
        d = np.repeat(np.asarray(defenders), len(scorers))
        s = np.tile(np.asarray(scorers), len(defenders))
        return self.predict_pairs(s, d).reshape(len(defenders), len(scorers))

    def total_effects(self, ref_scorers=None, ref_w=None, ref_defenders=None, ref_dw=None) -> pd.DataFrame:
        """Each player's full effect, size terms included, as a % change
        relative to the *possession-weighted average player* (the typical
        player actually on the floor), not a skill-0 player.

        defender_pct: points his man scores vs. the average defender, averaged
                      over a reference mix of scorers (negative = better).
        scorer_pct:   points he scores vs. the average scorer, averaged over a
                      reference mix of defenders (positive = better).
        Reference mixes default to everyone, weighted by matchup possessions.
        """
        ids = self.player_ids
        if ref_scorers is None:
            ref_scorers, ref_w = ids, self.poss_off
        if ref_defenders is None:
            ref_defenders, ref_dw = ids, self.poss_def
        ref_w = np.asarray(ref_w, float) / np.sum(ref_w)
        ref_dw = np.asarray(ref_dw, float) / np.sum(ref_dw)

        def_val, off_val = np.empty(len(ids)), np.empty(len(ids))
        for start in range(0, len(ids), 200):          # chunked to limit memory
            chunk = ids[start:start + 200]
            def_val[start:start + 200] = self.matrix(chunk, ref_scorers) @ ref_w
            off_val[start:start + 200] = self.matrix(ref_defenders, chunk).T @ ref_dw
        # baselines: what the average defender allows / average scorer scores
        base_def = float(np.sum(def_val * self.poss_def) / np.sum(self.poss_def))
        base_off = float(np.sum(off_val * self.poss_off) / np.sum(self.poss_off))
        return pd.DataFrame({"player_id": ids,
                             "defender_pct": 100 * (def_val / base_def - 1),
                             "scorer_pct": 100 * (off_val / base_off - 1)})

    def params(self) -> dict:
        p = {"intercept": self.intercept, "alpha": self.alpha,
             "poisson": 1.0 if self.kind == "poisson" else 0.0}
        p.update({f"w_{n}": float(w) for n, w in zip(FEATURES, self.feat_coef)})
        p.update(self.meta)
        return p


class PoissonRidgeNewton:
    """L2-penalized Poisson regression fit by Newton's method (IRLS).

    Minimizes the same objective as sklearn's PoissonRegressor:
        (1 / (2 * sum(w))) * sum(w * deviance) + (alpha / 2) * ||coef||^2
    i.e. sum(w * (mu - y * eta)) + (alpha * sum(w) / 2) * ||coef||^2,
    with an unpenalized intercept.

    Each Newton step solves a (p+1) x (p+1) linear system (p ~ 1,700 here),
    built from the sparse design, so it converges in a handful of steps where
    the general-purpose L-BFGS solver crawls on this badly conditioned problem.
    """

    def __init__(self, alpha: float, max_iter: int = 100, tol: float = 1e-9):
        self.alpha, self.max_iter, self.tol = alpha, max_iter, tol

    @staticmethod
    def _objective(eta, y, w, coef, lam):
        return float(np.sum(w * (np.exp(eta) - y * eta)) + 0.5 * lam * coef @ coef)

    def fit(self, X, y, sample_weight):
        X = sp.csr_matrix(X)
        w = np.asarray(sample_weight, float)
        y = np.asarray(y, float)
        n, p = X.shape
        lam = self.alpha * w.sum()
        coef = np.zeros(p)
        b0 = float(np.log(max(np.sum(w * y) / w.sum(), 1e-12)))
        eta = np.full(n, b0)
        obj = self._objective(eta, y, w, coef, lam)
        self.n_iter_ = 0
        for it in range(1, self.max_iter + 1):
            mu = np.exp(eta)
            r = w * (mu - y)                          # gradient wrt eta
            wm = w * mu                               # Hessian weights
            g = np.concatenate([[r.sum()], X.T @ r + lam * coef])
            XtW = X.T.multiply(wm).tocsr()            # p x n
            H = np.empty((p + 1, p + 1))
            H[0, 0] = wm.sum()
            H[0, 1:] = H[1:, 0] = np.asarray(XtW.sum(axis=1)).ravel()
            H[1:, 1:] = (XtW @ X).toarray()
            H[1:, 1:][np.diag_indices(p)] += lam
            step = np.linalg.solve(H, g)
            t = 1.0                                   # backtracking line search
            while True:
                new_b0, new_coef = b0 - t * step[0], coef - t * step[1:]
                new_eta = new_b0 + X @ new_coef
                new_obj = self._objective(new_eta, y, w, new_coef, lam)
                if np.isfinite(new_obj) and new_obj <= obj + 1e-12 * abs(obj):
                    break
                t *= 0.5
                if t < 1e-8:
                    break
            b0, coef, eta = new_b0, new_coef, new_eta
            done = abs(obj - new_obj) <= self.tol * max(1.0, abs(obj)) or np.max(np.abs(t * step)) < 1e-10
            obj = new_obj
            self.n_iter_ = it
            if done:
                break
        self.intercept_, self.coef_ = b0, coef
        return self


def _estimator(kind: str, alpha: float):
    if kind == "poisson":
        return PoissonRidgeNewton(alpha=alpha)
    return Ridge(alpha=alpha, fit_intercept=True, solver="sparse_cg", tol=1e-6, max_iter=5000)


def _target(kind: str, design: MatchupDesign) -> np.ndarray:
    # Poisson models the rate per possession; linear models pts per 100 directly.
    return design.rate if kind == "poisson" else 100.0 * design.rate


def _columns(design: MatchupDesign, which: str) -> np.ndarray:
    """which: 'scorer' | 'scorer+defender' | 'full' (players + size/style)
    | 'full+context' (also team-season context, if the design has it)."""
    P = design.n_players
    return {"scorer": np.arange(P),
            "scorer+defender": np.arange(2 * P),
            "full": np.arange(2 * P + len(FEATURES))}.get(which, np.arange(design.X.shape[1]))


def fit(design: MatchupDesign, feats, meta, alpha: float, kind: str = "poisson",
        which: str = "full+context") -> MatchupModel:
    cols = _columns(design, which)
    est = _estimator(kind, alpha).fit(design.X[:, cols], _target(kind, design), sample_weight=design.w)
    coef = np.zeros(design.X.shape[1])
    coef[cols] = est.coef_
    P, k = design.n_players, len(FEATURES)
    poss_by_col = design.X[:, :2 * P].T @ design.poss
    return MatchupModel(
        kind=kind, alpha=alpha, intercept=float(est.intercept_), player_ids=design.player_ids,
        scorer=coef[:P], defender=coef[P:2 * P], feat_coef=coef[2 * P:2 * P + k] * FEATURE_SCALE,
        feats=feats, meta=meta, poss_off=poss_by_col[:P], poss_def=poss_by_col[P:],
        ctx_keys=design.ctx_keys, ctx_coef=coef[2 * P + k:])


# ---------------------------------------------------------- evaluation ----

def weighted_rmse(y, pred, w) -> float:
    return float(np.sqrt(np.sum(w * (y - pred) ** 2) / np.sum(w)))


def poisson_deviance(pts, poss, pred_per100) -> float:
    """Mean Poisson deviance per possession (lower = better)."""
    mu = np.maximum(pred_per100 / 100.0 * poss, 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(pts > 0, pts * np.log(pts / mu), 0.0)
    return float(2 * np.sum(term - (pts - mu)) / np.sum(poss))


def score(model: MatchupModel, rows: pd.DataFrame) -> dict:
    """Score a model on raw (per-game) rows."""
    pred = model.predict_rows(rows)
    poss = rows["partial_poss"].to_numpy(dtype=float)
    pts = rows["player_pts"].to_numpy(dtype=float)
    return {"rmse": weighted_rmse(100 * pts / poss, pred, poss),
            "deviance": poisson_deviance(pts, poss, pred)}


def cross_validate(rows: pd.DataFrame, feats, meta, weights, kind: str = "poisson",
                   alphas=None, n_splits: int = 5, which: str = "full+context", log=print):
    """Pick alpha with GroupKFold by game on raw rows. Each training fold is
    aggregated before fitting; scoring is on the held-out fold's raw rows.
    Poisson is scored by deviance, linear by RMSE."""
    alphas = ALPHAS[kind] if alphas is None else alphas
    metric = "deviance" if kind == "poisson" else "rmse"
    player_ids = np.union1d(rows["off_player_id"].unique(), rows["def_player_id"].unique())
    ctx_cols = [c for c in CTX_COLUMNS if c in rows.columns]
    ctx_keys = np.unique(np.concatenate([rows[c].unique() for c in ctx_cols])) if ctx_cols else None
    folds = list(GroupKFold(n_splits=n_splits).split(rows, groups=rows["game_id"]))
    train_designs = [build(aggregate(rows.iloc[tr]), feats, meta, weights, player_ids, ctx_keys)
                     for tr, _ in folds]
    out = []
    for a in alphas:
        vals = [score(fit(d, feats, meta, a, kind, which), rows.iloc[te])[metric]
                for d, (_, te) in zip(train_designs, folds)]
        out.append({"alpha": float(a), metric: float(np.mean(vals))})
        log(f"  [{kind}, {which}] alpha={a:>11.3g}  cv_{metric}={np.mean(vals):.5f}")
    table = pd.DataFrame(out)
    best = table.loc[table[metric].idxmin()]
    if best["alpha"] in (table["alpha"].min(), table["alpha"].max()):
        log("  WARNING: best alpha is at the edge of the grid; consider widening it.")
    return float(best["alpha"]), table
