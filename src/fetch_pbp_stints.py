"""Turn each game's play-by-play into stints (same 10 players on the floor).

Uses pbpstats with the 'live' data provider (NBA's liveData feed). The old
stats.nba.com playbyplayv2 endpoint now returns empty data, so the default
pbpstats provider no longer works.
"""
from dataclasses import dataclass

from pbpstats.client import Client
from pbpstats.resources.enhanced_pbp import FieldGoal, FreeThrow

import config
from src import db
from src.utils import get_logger

log = get_logger("stints")


@dataclass
class PossessionRecord:
    period: int
    offense_team_id: int
    home_players: tuple   # sorted player IDs
    away_players: tuple
    seconds: float
    home_pts: int
    away_pts: int


def _client(game_id: str) -> Client:
    """Use cached files when we already downloaded this game, else the web."""
    d = config.PBP_DIR
    cached = ((d / "pbp" / f"live_{game_id}.json").exists()
              and (d / "game_details" / f"live_{game_id}.json").exists())
    source = "file" if cached else "web"
    # pbpstats writes its cache into these subfolders but doesn't create them
    for sub in ("pbp", "game_details", "schedule"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return Client({
        "dir": str(d),
        "Boxscore": {"source": source, "data_provider": "live"},
        "Possessions": {"source": source, "data_provider": "live"},
    })


def _period_length(period: int) -> float:
    return 720.0 if period <= 4 else 300.0


def extract_possessions(game_id: str, home_team_id: int, away_team_id: int) -> tuple[list, int]:
    """Download/parse one game. Returns (possession records, # skipped)."""
    game = _client(game_id).Game(game_id)
    records, skipped = [], 0
    prev_period, prev_remaining = None, None

    for poss in game.possessions.items:
        if not poss.events:
            continue
        period = poss.events[0].period
        if period != prev_period:
            prev_period, prev_remaining = period, _period_length(period)
        end_remaining = float(poss.events[-1].seconds_remaining)
        seconds = max(prev_remaining - end_remaining, 0.0)
        prev_remaining = end_remaining

        # Lineups at the start of the possession
        players = poss.events[0].current_players
        home = tuple(sorted(players.get(home_team_id, [])))
        away = tuple(sorted(players.get(away_team_id, [])))
        if len(home) != 5 or len(away) != 5:
            skipped += 1
            continue

        # Credit every made shot / free throw to the team that made it
        pts = {home_team_id: 0, away_team_id: 0}
        for ev in poss.events:
            team = getattr(ev, "team_id", None)
            if team not in pts:
                continue
            if isinstance(ev, FieldGoal) and ev.is_made:
                pts[team] += ev.shot_value
            elif isinstance(ev, FreeThrow) and ev.is_made:
                pts[team] += 1

        records.append(PossessionRecord(
            period=period, offense_team_id=poss.offense_team_id,
            home_players=home, away_players=away, seconds=seconds,
            home_pts=pts[home_team_id], away_pts=pts[away_team_id]))
    return records, skipped


def possessions_to_stints(records: list[PossessionRecord], home_team_id: int) -> list[dict]:
    """Merge consecutive possessions with the same 10 players (in the same
    period) into stints."""
    stints = []
    for r in records:
        key = (r.period, r.home_players, r.away_players)
        if stints and stints[-1]["key"] == key:
            s = stints[-1]
        else:
            s = {"key": key, "period": r.period, "home_players": list(r.home_players),
                 "away_players": list(r.away_players), "seconds": 0.0,
                 "home_poss": 0, "away_poss": 0, "home_pts": 0, "away_pts": 0}
            stints.append(s)
        s["seconds"] += r.seconds
        if r.offense_team_id == home_team_id:
            s["home_poss"] += 1
        else:
            s["away_poss"] += 1
        s["home_pts"] += r.home_pts
        s["away_pts"] += r.away_pts
    return stints


def load_game(conn, game_id: str, home_team_id: int, away_team_id: int) -> int:
    records, skipped = extract_possessions(game_id, home_team_id, away_team_id)
    stints = possessions_to_stints(records, home_team_id)
    rows = [(game_id, i, s["period"], s["home_players"], s["away_players"],
             round(s["seconds"], 1), s["home_poss"], s["away_poss"],
             s["home_pts"], s["away_pts"])
            for i, s in enumerate(stints, start=1)]
    with conn.cursor() as cur:  # replace any partial earlier load
        cur.execute("DELETE FROM stints WHERE game_id = %s", (game_id,))
    db.upsert(conn, "stints",
              ["game_id", "stint_num", "period", "home_players", "away_players",
               "seconds", "home_poss", "away_poss", "home_pts", "away_pts"],
              rows, conflict=["game_id", "stint_num"])
    if skipped:
        log.debug("%s: skipped %d possessions with incomplete lineups", game_id, skipped)
    return len(rows)
