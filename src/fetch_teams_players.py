"""Load teams (static list) and player bios (height, weight, position)."""
from nba_api.stats.endpoints import playerindex
from nba_api.stats.static import teams as static_teams

import config
from src import db
from src.utils import get_logger, height_to_inches, to_int, with_retries

log = get_logger("teams_players")


def load_teams(conn) -> int:
    rows = [(t["id"], t["abbreviation"], t["city"], t["nickname"])
            for t in static_teams.get_teams()]
    n = db.upsert(conn, "teams", ["team_id", "abbreviation", "city", "name"], rows,
                  conflict=["team_id"], update=["abbreviation", "city", "name"])
    log.info("Loaded %d teams", n)
    return n


def load_player_bios(conn, season: str) -> int:
    """PlayerIndex with historical=1 returns every player with bio info in one call."""
    resp = with_retries(playerindex.PlayerIndex, season=season, historical_nullable="1",
                        timeout=config.REQUEST_TIMEOUT)
    df = resp.get_data_frames()[0]
    rows = []
    for r in df.itertuples(index=False):
        name = f"{r.PLAYER_FIRST_NAME} {r.PLAYER_LAST_NAME}".strip()
        rows.append((int(r.PERSON_ID), name, r.POSITION or None,
                     height_to_inches(r.HEIGHT), to_int(r.WEIGHT)))
    n = db.upsert(conn, "players",
                  ["player_id", "full_name", "position", "height_inches", "weight_lbs"],
                  rows, conflict=["player_id"],
                  update=["full_name", "position", "height_inches", "weight_lbs"])
    log.info("Loaded %d player bios", n)
    return n
