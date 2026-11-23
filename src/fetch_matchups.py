"""Who guarded whom, per game (NBA.com BoxScoreMatchupsV3)."""
from nba_api.stats.endpoints import boxscorematchupsv3

import config
from src import db
from src.utils import clock_to_seconds, get_logger, read_json, to_int, with_retries, write_json

log = get_logger("matchups")

COLUMNS = ["game_id", "off_team_id", "off_player_id", "def_player_id", "matchup_seconds",
           "partial_poss", "player_pts", "team_pts", "fgm", "fga", "fg3m", "fg3a",
           "ftm", "fta", "ast", "tov", "blk", "help_blk", "switches_on", "shooting_fouls"]


def _fetch(game_id: str) -> list[dict]:
    cache = config.MATCHUP_DIR / f"{game_id}.json"
    data = read_json(cache)
    if data is None:
        resp = with_retries(boxscorematchupsv3.BoxScoreMatchupsV3, game_id=game_id,
                            timeout=config.REQUEST_TIMEOUT)
        data = resp.get_data_frames()[0].to_dict("records")
        write_json(cache, data)
    return data


def row_from_record(game_id: str, r: dict) -> tuple:
    return (
        game_id,
        to_int(r.get("teamId")),
        to_int(r.get("personIdOff")),
        to_int(r.get("personIdDef")),
        clock_to_seconds(r.get("matchupMinutes")),
        float(r.get("partialPossessions") or 0),
        to_int(r.get("playerPoints")),
        to_int(r.get("teamPoints")),
        to_int(r.get("matchupFieldGoalsMade")),
        to_int(r.get("matchupFieldGoalsAttempted")),
        to_int(r.get("matchupThreePointersMade")),
        to_int(r.get("matchupThreePointersAttempted")),
        to_int(r.get("matchupFreeThrowsMade")),
        to_int(r.get("matchupFreeThrowsAttempted")),
        to_int(r.get("matchupAssists")),
        to_int(r.get("matchupTurnovers")),
        to_int(r.get("matchupBlocks")),
        to_int(r.get("helpBlocks")),
        to_int(r.get("switchesOn")),
        to_int(r.get("shootingFouls")),
    )


def load_game(conn, game_id: str) -> int:
    rows = [row_from_record(game_id, r) for r in _fetch(game_id)]
    rows = [r for r in rows if r[2] is not None and r[3] is not None]
    db.upsert(conn, "matchups", COLUMNS, rows,
              conflict=["game_id", "off_player_id", "def_player_id"],
              update=COLUMNS[4:])
    return len(rows)
