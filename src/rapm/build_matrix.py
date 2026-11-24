"""Turn stints into the RAPM design matrix.

Each stint becomes two rows, one per team on offense:

    target  = points scored per 100 possessions
    weight  = possessions x season recency weight
    columns = +1 for each of the 5 offensive players' O column
              +1 for each of the 5 defensive players' D column
              +1 in the home-court column when the home team has the ball

Column layout: [O_0 .. O_{P-1} | D_0 .. D_{P-1} | home]
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class Design:
    X: sp.csr_matrix
    y: np.ndarray            # points per 100 possessions
    w: np.ndarray            # possessions x recency weight
    poss: np.ndarray         # raw possessions (unweighted)
    groups: np.ndarray       # game_id per row, for game-level CV splits
    player_ids: np.ndarray   # player id for column j (and P + j)

    @property
    def n_players(self) -> int:
        return len(self.player_ids)

    @property
    def home_col(self) -> int:
        return 2 * self.n_players


def load_stints(conn, seasons: list[str]) -> pd.DataFrame:
    sql = """
        SELECT s.game_id, g.season, s.home_players, s.away_players,
               s.home_poss, s.away_poss, s.home_pts, s.away_pts
        FROM stints s JOIN games g USING (game_id)
        WHERE g.season = ANY(%s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (list(seasons),))
        cols = [c[0] for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def season_weights(seasons: list[str], decay: float = 0.5) -> dict[str, float]:
    """Most recent season gets 1.0, each earlier season is multiplied by decay."""
    ordered = sorted(seasons, reverse=True)
    return {s: decay ** i for i, s in enumerate(ordered)}


def _to_rows(stints: pd.DataFrame) -> pd.DataFrame:
    """Two rows per stint: home on offense, away on offense."""
    home = pd.DataFrame({
        "game_id": stints["game_id"], "season": stints["season"],
        "offense": stints["home_players"], "defense": stints["away_players"],
        "poss": stints["home_poss"], "pts": stints["home_pts"], "home": 1,
    })
    away = pd.DataFrame({
        "game_id": stints["game_id"], "season": stints["season"],
        "offense": stints["away_players"], "defense": stints["home_players"],
        "poss": stints["away_poss"], "pts": stints["away_pts"], "home": 0,
    })
    rows = pd.concat([home, away], ignore_index=True)
    return rows[rows["poss"] > 0].reset_index(drop=True)


def build(stints: pd.DataFrame, weights: dict[str, float] | None = None,
          player_ids: np.ndarray | None = None) -> Design:
    """Build the design matrix.

    player_ids: reuse an existing column layout (e.g. the training set's when
    building a test set). Players not in it are left out of the row, which is
    the same as treating them as exactly average.
    """
    rows = _to_rows(stints)
    off = np.array(rows["offense"].tolist(), dtype=np.int64)
    dfn = np.array(rows["defense"].tolist(), dtype=np.int64)

    if player_ids is None:
        player_ids = np.unique(np.concatenate([off.ravel(), dfn.ravel()]))
    player_ids = np.asarray(player_ids, dtype=np.int64)
    n_players, n_rows = len(player_ids), len(rows)

    def col_index(ids: np.ndarray):
        pos = np.searchsorted(player_ids, ids)
        pos = np.clip(pos, 0, n_players - 1)
        known = player_ids[pos] == ids
        return pos, known

    off_pos, off_known = col_index(off)
    def_pos, def_known = col_index(dfn)
    row_idx = np.repeat(np.arange(n_rows), 5).reshape(n_rows, 5)

    home = rows["home"].to_numpy()
    r = np.concatenate([row_idx[off_known], row_idx[def_known], np.flatnonzero(home)])
    c = np.concatenate([off_pos[off_known], n_players + def_pos[def_known],
                        np.full(int(home.sum()), 2 * n_players)])
    X = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n_rows, 2 * n_players + 1))

    poss = rows["poss"].to_numpy(dtype=float)
    y = 100.0 * rows["pts"].to_numpy(dtype=float) / poss
    if weights:
        season_w = rows["season"].map(weights).fillna(0.0).to_numpy(dtype=float)
    else:
        season_w = np.ones(n_rows)
    return Design(X=X, y=y, w=poss * season_w, poss=poss,
                  groups=rows["game_id"].to_numpy(), player_ids=player_ids)
