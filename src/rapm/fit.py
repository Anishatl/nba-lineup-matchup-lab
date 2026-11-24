"""Fit RAPM: weighted ridge regression with the penalty chosen by game-level CV.

Objective (scikit-learn Ridge with sample_weight):

    sum_i w_i * (y_i - b0 - x_i . beta)^2  +  alpha * ||beta||^2

The intercept b0 is never penalized. The home-court column shouldn't be
either, so it's multiplied by HOME_SCALE before fitting: its coefficient
shrinks by HOME_SCALE, so the penalty on it becomes negligible
(alpha / HOME_SCALE^2). It's scaled back afterwards.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from src.rapm.build_matrix import Design

HOME_SCALE = 100.0
DEFAULT_ALPHAS = np.logspace(2, 4.5, 11)   # 100 ... ~31,600
SOLVER = "sparse_cg"


@dataclass
class RapmModel:
    alpha: float
    intercept: float
    home_adv: float
    player_ids: np.ndarray
    o: np.ndarray            # offensive rating per player (positive = good)
    d: np.ndarray            # defensive rating per player (negative = good)

    def coef_vector(self) -> np.ndarray:
        return np.concatenate([self.o, self.d, [self.home_adv]])

    def predict(self, design: Design) -> np.ndarray:
        """Predicted pts per 100 for each row. `design` must use this model's
        player_ids layout (build(..., player_ids=model.player_ids))."""
        return self.intercept + design.X @ self.coef_vector()

    def ratings_frame(self, design: Design | None = None) -> pd.DataFrame:
        df = pd.DataFrame({"player_id": self.player_ids, "o_rapm": self.o, "d_rapm": self.d})
        df["total_rapm"] = df["o_rapm"] - df["d_rapm"]
        if design is not None:
            P = len(self.player_ids)
            poss_by_col = design.X.T @ design.poss   # possessions per column
            df["off_poss"] = np.rint(poss_by_col[:P]).astype(int)
            df["def_poss"] = np.rint(poss_by_col[P:2 * P]).astype(int)
        return df


def _scaled(X: sp.csr_matrix, home_col: int) -> sp.csr_matrix:
    scale = np.ones(X.shape[1])
    scale[home_col] = HOME_SCALE
    return (X @ sp.diags(scale)).tocsr()


def weighted_rmse(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    return float(np.sqrt(np.sum(w * (y - pred) ** 2) / np.sum(w)))


def _ridge(alpha: float) -> Ridge:
    return Ridge(alpha=alpha, fit_intercept=True, solver=SOLVER, tol=1e-6, max_iter=5000)


def cross_validate(design: Design, alphas=DEFAULT_ALPHAS, n_splits: int = 5,
                   log=print) -> tuple[float, pd.DataFrame]:
    """Pick alpha by GroupKFold on game_id: every stint from a game lands in
    the same fold, so the model is always scored on games it never saw."""
    X = _scaled(design.X, design.home_col)
    folds = list(GroupKFold(n_splits=n_splits).split(X, design.y, design.groups))
    results = []
    for alpha in alphas:
        errs = []
        for train, test in folds:
            m = _ridge(alpha).fit(X[train], design.y[train], sample_weight=design.w[train])
            errs.append(weighted_rmse(design.y[test], m.predict(X[test]), design.w[test]))
        results.append({"alpha": float(alpha), "cv_rmse": float(np.mean(errs))})
        log(f"  alpha={alpha:>9.1f}  cv_rmse={np.mean(errs):.4f}")
    table = pd.DataFrame(results)
    best = table.loc[table["cv_rmse"].idxmin()]
    if best["alpha"] in (table["alpha"].min(), table["alpha"].max()):
        log("  WARNING: best alpha is at the edge of the grid; consider widening it.")
    return float(best["alpha"]), table


def fit(design: Design, alpha: float) -> RapmModel:
    X = _scaled(design.X, design.home_col)
    m = _ridge(alpha).fit(X, design.y, sample_weight=design.w)
    coef = m.coef_.copy()
    coef[design.home_col] *= HOME_SCALE
    P = design.n_players
    return RapmModel(alpha=alpha, intercept=float(m.intercept_),
                     home_adv=float(coef[design.home_col]),
                     player_ids=design.player_ids, o=coef[:P], d=coef[P:2 * P])
