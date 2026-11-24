"""Save a fitted RAPM model to model_runs + player_ratings."""
import numpy as np
import pandas as pd

from src import db
from src.rapm.build_matrix import Design
from src.rapm.fit import RapmModel


def save_run(conn, model: RapmModel, design: Design, seasons: list[str],
             weights: dict[str, float], cv_rmse: float, notes: str = "") -> int:
    ordered = sorted(seasons)
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO model_runs (model, seasons, season_weights, alpha, intercept,
                                       home_adv, n_rows, n_players, cv_rmse, notes)
               VALUES ('rapm', %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING run_id""",
            (ordered, [weights[s] for s in ordered], model.alpha, model.intercept,
             model.home_adv, int(len(design.y)), int(design.n_players), cv_rmse, notes))
        run_id = cur.fetchone()[0]

    ratings = model.ratings_frame(design)
    rows = [(run_id, int(r.player_id), float(r.o_rapm), float(r.d_rapm), float(r.total_rapm),
             int(r.off_poss), int(r.def_poss)) for r in ratings.itertuples(index=False)]
    db.upsert(conn, "player_ratings",
              ["run_id", "player_id", "o_rapm", "d_rapm", "total_rapm", "off_poss", "def_poss"],
              rows, conflict=["run_id", "player_id"])
    return run_id


def named_ratings(conn, ratings: pd.DataFrame) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute("SELECT player_id, full_name FROM players WHERE player_id = ANY(%s)",
                    ([int(p) for p in ratings["player_id"]],))
        names = dict(cur.fetchall())
    out = ratings.copy()
    out.insert(1, "name", out["player_id"].map(names).fillna("?"))
    out["d_display"] = -out["d_rapm"]   # flipped for reading: higher = better defense
    return out.replace([np.inf, -np.inf], np.nan)
