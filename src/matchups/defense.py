"""Calibrated defensive impact: how many points per 100 possessions a player
saves his team, using the weights the held-out calibration assigned to each
source of defensive information.

    points allowed per 100 = ... + b * (matchup defense) + c * sum(D-RAPM) + ...

So one player's contribution to points allowed is

    c * D-RAPM  +  b * (his matchup effect, pts per 100 matchup possessions)

and "points saved" is the negative of that. D-RAPM comes from real on-floor
points allowed and was validated out of sample; matchup defense counts only as
much as calibration says it deserves (b is ~1/4 of c on real data).

Flags (among qualified players) point out where the two sources disagree:
    switched_off  matchup numbers in the top 20% but D-RAPM in the bottom 40%:
                  likely targeted on screens, with the points credited to
                  whoever switched onto his man
    help_rim      D-RAPM in the top 20% but matchup numbers only average or
                  worse: value comes from help / rim protection, not one-on-one
"""
import numpy as np
import pandas as pd

from src.matchups.model import MatchupModel
from src.matchups.optimize import Calibration


def defensive_impact(model: MatchupModel, d_rapm: pd.Series, cal: Calibration,
                     min_poss: float = 1500) -> pd.DataFrame:
    eff = model.total_effects()
    eff["def_poss"] = model.poss_def
    eff["d_rapm"] = d_rapm.reindex(eff["player_id"]).to_numpy(dtype=float)
    eff["d_rapm"] = eff["d_rapm"].fillna(0.0)
    allowed = cal.c * eff["d_rapm"] + cal.b * eff["defender_pts"]
    eff["pts_saved"] = -allowed                       # per 100 possessions, higher = better
    eff["from_rapm"] = -cal.c * eff["d_rapm"]
    eff["from_matchup"] = -cal.b * eff["defender_pts"]

    q = eff["def_poss"] >= min_poss
    eff["flag"] = ""
    if q.sum() >= 10:
        m_pct = (-eff.loc[q, "defender_pts"]).rank(pct=True)   # 1 = best one-on-one
        r_pct = (-eff.loc[q, "d_rapm"]).rank(pct=True)         # 1 = best D-RAPM
        eff.loc[m_pct[(m_pct >= 0.8) & (r_pct <= 0.4)].index, "flag"] = "switched_off"
        eff.loc[r_pct[(r_pct >= 0.8) & (m_pct <= 0.5)].index, "flag"] = "help_rim"
    return eff
