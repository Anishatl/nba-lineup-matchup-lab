"""Combine RAPM offense + matchup-model defense into lineup scores, and
optimize assignments with the Hungarian algorithm.

Calibrated scoring model (fit in calibrate.py on a held-out season):

    pts/100 scored by lineup A against lineup B
        = base + a * sum(O-RAPM of A)
               + b * (best matchup defense of B on A  -  dm_mean)
               + c * sum(D-RAPM of B)

    "best matchup defense of B on A" = the Hungarian-optimal assignment of B's
    5 defenders onto A's 5 scorers, summing predicted pts/100 per matchup.

Lineup value for team A against opponent B, with the defensive-emphasis slider:
    offense = scored(A vs B) - base
    defense = scored(B vs A) - base          (points allowed above baseline)
    net     = offense - slider * defense     (slider = 1 -> plain net rating)

Every function takes plain lists of player IDs, so any players can be used on
either side.
"""
from dataclasses import dataclass
from itertools import combinations, permutations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.matchups.model import MatchupModel


@dataclass
class Calibration:
    base: float
    a: float          # weight on sum of offensive RAPM
    b: float          # weight on matchup-model defense
    c: float          # weight on sum of defensive RAPM (help defense etc.)
    dm_mean: float    # average optimal matchup defense (centering constant)


@dataclass
class Engine:
    matchup: MatchupModel
    o_rapm: pd.Series          # indexed by player_id
    d_rapm: pd.Series          # raw coefficients: negative = good defense
    cal: Calibration

    # --- building blocks ---------------------------------------------------
    def _sum(self, series: pd.Series, ids) -> float:
        return float(series.reindex(list(ids)).fillna(0.0).sum())

    def best_assignment(self, defenders, scorers):
        """Hungarian algorithm. Works for 5x5 and for N defenders x 5 scorers
        (then it also picks which defenders to use). Returns
        (list of (defender, scorer, predicted pts/100), total)."""
        M = self.matchup.matrix(defenders, scorers)
        rows, cols = linear_sum_assignment(M)
        pairs = [(defenders[i], scorers[j], float(M[i, j])) for i, j in zip(rows, cols)]
        return pairs, float(M[rows, cols].sum())

    def scored(self, offense, defense) -> float:
        """Calibrated pts/100 for lineup `offense` against lineup `defense`."""
        _, dm = self.best_assignment(list(defense), list(offense))
        c = self.cal
        return (c.base + c.a * self._sum(self.o_rapm, offense)
                + c.b * (dm - c.dm_mean) + c.c * self._sum(self.d_rapm, defense))

    def top_assignments(self, defenders, scorers, k: int = 2):
        """The k best one-to-one assignments of 5 defenders onto 5 scorers
        (all 120 are checked). Returns [(pairs, total), ...], best first."""
        defenders, scorers = list(defenders), list(scorers)
        M = self.matchup.matrix(defenders, scorers)
        if len(defenders) != len(scorers):
            return [self.best_assignment(defenders, scorers)]
        results = []
        for perm in permutations(range(len(scorers))):
            total = float(sum(M[i, j] for i, j in enumerate(perm)))
            results.append(([(defenders[i], scorers[j], float(M[i, j])) for i, j in enumerate(perm)], total))
        results.sort(key=lambda r: r[1])
        return results[:k]

    def evaluate(self, mine, theirs, assignment: dict, slider: float = 1.0) -> dict:
        """Value of a user-chosen setup. `assignment` maps each of my players
        to the opponent he guards. Points allowed use that assignment; points
        scored assume the opponent defends my five optimally."""
        mine, theirs = list(mine), list(theirs)
        if sorted(assignment) != sorted(mine) or sorted(assignment.values()) != sorted(theirs):
            raise ValueError("assignment must map each of my 5 players to a different opponent")
        c = self.cal
        pairs = [(d, s, float(self.matchup.predict_pairs([s], [d])[0])) for d, s in assignment.items()]
        dm_allowed = sum(p[2] for p in pairs)
        allowed = (c.base + c.a * self._sum(self.o_rapm, theirs)
                   + c.b * (dm_allowed - c.dm_mean) + c.c * self._sum(self.d_rapm, mine))
        scored = self.scored(mine, theirs)
        best = self.lineup_value(mine, theirs, slider)
        net = (scored - c.base) - slider * (allowed - c.base)
        return {"lineup": tuple(mine), "scored": scored, "allowed": allowed, "net": net,
                "assignment": pairs, "best_net": best["net"], "best_assignment": best["assignment"],
                "cost_vs_best": best["net"] - net}

    # --- user-facing -------------------------------------------------------
    def lineup_value(self, mine, theirs, slider: float = 1.0) -> dict:
        scored = self.scored(mine, theirs)
        allowed = self.scored(theirs, mine)
        assignment, _ = self.best_assignment(list(mine), list(theirs))
        off, dfn = scored - self.cal.base, allowed - self.cal.base
        return {"lineup": tuple(mine), "scored": scored, "allowed": allowed,
                "net": off - slider * dfn, "assignment": assignment}

    def rank_lineups(self, roster, theirs, slider: float = 1.0, top: int = 10) -> list[dict]:
        """Try every 5-man combination of `roster` against `theirs`.
        Pair predictions are computed once up front, then sliced per lineup."""
        roster, theirs = list(roster), list(theirs)
        c = self.cal
        m_def = self.matchup.matrix(roster, theirs)     # our i guards their j
        m_off = self.matchup.matrix(theirs, roster)     # their i guards our j
        o_ours = self.o_rapm.reindex(roster).fillna(0.0).to_numpy()
        d_ours = self.d_rapm.reindex(roster).fillna(0.0).to_numpy()
        o_theirs = self._sum(self.o_rapm, theirs)
        d_theirs = self._sum(self.d_rapm, theirs)

        results = []
        for idx in combinations(range(len(roster)), 5):
            idx = list(idx)
            sub_def = m_def[idx, :]
            r, k = linear_sum_assignment(sub_def)
            dm_allowed = sub_def[r, k].sum()
            sub_off = m_off[:, idx]
            r2, k2 = linear_sum_assignment(sub_off)
            dm_scored = sub_off[r2, k2].sum()

            scored = c.base + c.a * o_ours[idx].sum() + c.b * (dm_scored - c.dm_mean) + c.c * d_theirs
            allowed = c.base + c.a * o_theirs + c.b * (dm_allowed - c.dm_mean) + c.c * d_ours[idx].sum()
            off, dfn = scored - c.base, allowed - c.base
            results.append({
                "lineup": tuple(roster[i] for i in idx), "scored": scored, "allowed": allowed,
                "net": off - slider * dfn,
                "assignment": [(roster[idx[i]], theirs[j], float(sub_def[i, j])) for i, j in zip(r, k)],
            })
        return sorted(results, key=lambda x: x["net"], reverse=True)[:top]
