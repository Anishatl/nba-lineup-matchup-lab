"""Save the matchup model to Postgres and load a ready-to-use Engine back.

Loading only needs stored numbers (player skills, features, scalar params),
which is exactly what the static dashboard will get as JSON.
"""
import numpy as np
import pandas as pd

from src import db
from src.matchups import model as mm
from src.matchups.optimize import Calibration, Engine


def save(conn, model: mm.MatchupModel, seasons, weights, cv_rmse, n_rows,
         calibration=None, home=None, notes="") -> int:
    ordered = sorted(seasons)
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO model_runs (model, seasons, season_weights, alpha, intercept,
                                       n_rows, n_players, cv_rmse, notes)
               VALUES ('matchup', %s, %s, %s, %s, %s, %s, %s, %s) RETURNING run_id""",
            (ordered, [weights[s] for s in ordered], model.alpha, model.intercept,
             int(n_rows), int(len(model.player_ids)), cv_rmse, notes))
        run_id = cur.fetchone()[0]

    f = model.feats.reindex(model.player_ids)
    rows = [(run_id, int(p), float(s), float(d), float(po), float(pd_),
             float(h), float(w), float(t))
            for p, s, d, po, pd_, h, w, t in zip(
                model.player_ids, model.scorer, model.defender, model.poss_off, model.poss_def,
                f["height_inches"], f["weight_lbs"], f["three_rate"])]
    db.upsert(conn, "matchup_ratings",
              ["run_id", "player_id", "scorer_skill", "defender_skill", "off_poss", "def_poss",
               "height_inches", "weight_lbs", "three_rate"],
              rows, conflict=["run_id", "player_id"])

    params = model.params()
    if calibration is not None:
        params.update({f"cal_{k}": v for k, v in calibration.__dict__.items()})
        params["cal_home"] = home
    db.upsert(conn, "model_params", ["run_id", "name", "value"],
              [(run_id, k, float(v)) for k, v in params.items() if v is not None],
              conflict=["run_id", "name"], update=["value"])
    return run_id


def _latest_run(cur, model: str) -> int:
    cur.execute("SELECT MAX(run_id) FROM model_runs WHERE model = %s", (model,))
    run_id = cur.fetchone()[0]
    if run_id is None:
        raise RuntimeError(f"No '{model}' run found. Run the {model} script first.")
    return run_id


def load_engine(conn) -> Engine:
    """Latest RAPM run + latest matchup run (with its calibration)."""
    with conn.cursor() as cur:
        rapm_run = _latest_run(cur, "rapm")
        m_run = _latest_run(cur, "matchup")
        cur.execute("SELECT player_id, o_rapm, d_rapm FROM player_ratings WHERE run_id = %s", (rapm_run,))
        r = pd.DataFrame(cur.fetchall(), columns=["player_id", "o", "d"]).set_index("player_id").astype(float)
        cur.execute("""SELECT player_id, scorer_skill, defender_skill, off_poss, def_poss,
                              height_inches, weight_lbs, three_rate
                       FROM matchup_ratings WHERE run_id = %s ORDER BY player_id""", (m_run,))
        m = pd.DataFrame(cur.fetchall(), columns=["player_id", "scorer", "defender", "off_poss", "def_poss",
                                                  "height_inches", "weight_lbs", "three_rate"])
        cur.execute("SELECT name, value FROM model_params WHERE run_id = %s", (m_run,))
        p = {k: float(v) for k, v in cur.fetchall()}
    m = m.astype({c: float for c in m.columns if c != "player_id"})

    if "cal_a" not in p:
        raise RuntimeError("Latest matchup run has no calibration. Re-run scripts/run_matchups.py.")
    meta = {k: p[k] for k in ("height_mean", "weight_mean", "three_rate_mean")}
    feats = m.set_index("player_id")[["height_inches", "weight_lbs", "three_rate"]]
    model = mm.MatchupModel(
        alpha=p["alpha"], intercept=p["intercept"], player_ids=m["player_id"].to_numpy(np.int64),
        scorer=m["scorer"].to_numpy(), defender=m["defender"].to_numpy(),
        feat_coef=np.array([p[f"w_{n}"] for n in mm.FEATURES]), feats=feats, meta=meta,
        poss_off=m["off_poss"].to_numpy(), poss_def=m["def_poss"].to_numpy())
    cal = Calibration(base=p["cal_base"], a=p["cal_a"], b=p["cal_b"], c=p["cal_c"], dm_mean=p["cal_dm_mean"])
    return Engine(matchup=model, o_rapm=r["o"], d_rapm=r["d"], cal=cal)
