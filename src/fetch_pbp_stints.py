"""Turn each game's play-by-play into stints (same 10 players on the floor).

Uses pbpstats with the 'live' data provider (NBA's liveData feed). The old
stats.nba.com playbyplayv2 endpoint now returns empty data, so the default
pbpstats provider no longer works.
"""
import re
from dataclasses import dataclass

from nba_api.stats.endpoints import boxscoretraditionalv3
from pbpstats.client import Client
from pbpstats.resources.enhanced_pbp import FieldGoal, FreeThrow, InvalidNumberOfStartersException

import config
from src import db
from src import pbpstats_patches  # noqa: F401 - fixes newer live feed format
from src.utils import clock_to_seconds, get_logger, read_json, with_retries, write_json

log = get_logger("stints")

OVERRIDES_FILE = config.PBP_DIR / "overrides" / "missing_period_starters.json"
_STARTERS_ERR = re.compile(r"GameId: (\d+), Period: (\d+), TeamId: (\d+)")


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


def _period_start_tenths(period: int) -> int:
    """Game time at the start of a period, in tenths of a second (NBA.com range units)."""
    if period <= 4:
        return (period - 1) * 7200
    return 4 * 7200 + (period - 5) * 3000


def find_period_starters(game_id: str, period: int, team_id: int) -> list[int] | None:
    """Ask NBA.com's box score who played in the first few seconds of a period.
    Tries growing windows and returns the first window where the team has
    exactly 5 players with playing time."""
    start = _period_start_tenths(period)
    for window_sec in (3, 10, 30, 60):
        resp = with_retries(boxscoretraditionalv3.BoxScoreTraditionalV3,
                            game_id=game_id, range_type="2",
                            start_period="0", end_period="0",
                            start_range=str(start), end_range=str(start + window_sec * 10),
                            timeout=config.REQUEST_TIMEOUT)
        df = resp.get_data_frames()[0]
        played = [int(r.personId) for r in df.itertuples(index=False)
                  if int(r.teamId) == team_id and clock_to_seconds(r.minutes or 0) > 0]
        if len(played) == 5:
            return sorted(played)
        if len(played) > 5:  # a sub happened inside the window; smaller windows also failed
            break
    return None


def add_starter_override(game_id: str, period: int, team_id: int, players: list[int]) -> None:
    """Merge one fix into pbpstats' overrides file (it reads this automatically)."""
    data = read_json(OVERRIDES_FILE) or {}
    data.setdefault(game_id, {}).setdefault(str(period), {})[str(team_id)] = players
    write_json(OVERRIDES_FILE, data)


def _counted_points(events, home_team_id: int, away_team_id: int) -> dict:
    """Points from counting made shots / free throws (used only as a cross-check)."""
    pts = {home_team_id: 0, away_team_id: 0}
    for ev in events:
        team = getattr(ev, "team_id", None)
        if team not in pts:
            continue
        if isinstance(ev, FieldGoal) and ev.is_made:
            pts[team] += ev.shot_value
        elif isinstance(ev, FreeThrow) and ev.is_made:
            pts[team] += 1
    return pts


def extract_possessions(game_id: str, home_team_id: int, away_team_id: int) -> tuple[list, dict]:
    """Download/parse one game.

    Points come from the running score that NBA's live feed attaches to every
    event (home_score / away_score): points on a possession = score at its last
    event minus score at the previous possession's last event. This can't miss
    a basket the way counting individual makes can.

    Returns (possession records, diagnostics dict).
    """
    game = _client(game_id).Game(game_id)
    diag = {"possessions": 0, "filled": 0, "first_in_period_missing": 0,
            "skipped": 0, "skipped_home_pts": 0, "skipped_away_pts": 0,
            "counted_home_pts": 0, "counted_away_pts": 0, "final_home": 0, "final_away": 0}
    prev_period, prev_remaining = None, None
    score = (0, 0)  # running (home, away) score

    # Pass 1: timing, points, and whatever lineup info each possession has
    rows = []
    for poss in game.possessions.items:
        if not poss.events:
            continue
        diag["possessions"] += 1
        period = poss.events[0].period
        first_in_period = period != prev_period
        if first_in_period:
            prev_period, prev_remaining = period, _period_length(period)
        end_remaining = float(poss.events[-1].seconds_remaining)
        seconds = max(prev_remaining - end_remaining, 0.0)
        prev_remaining = end_remaining

        # Score change across this possession
        start = score
        for ev in poss.events:
            h, a = getattr(ev, "home_score", None), getattr(ev, "away_score", None)
            if isinstance(h, int) and isinstance(a, int) and (h, a) >= (0, 0):
                score = (max(score[0], h), max(score[1], a))  # score never goes down
        home_pts, away_pts = score[0] - start[0], score[1] - start[1]

        counted = _counted_points(poss.events, home_team_id, away_team_id)
        diag["counted_home_pts"] += counted[home_team_id]
        diag["counted_away_pts"] += counted[away_team_id]

        home = _lineup(poss.events, home_team_id)
        away = _lineup(poss.events, away_team_id)
        if first_in_period and (home is None or away is None):
            diag["first_in_period_missing"] += 1
        rows.append({"period": period, "offense": poss.offense_team_id, "seconds": seconds,
                     "home_pts": home_pts, "away_pts": away_pts, "home": home, "away": away})

    # Pass 2: fill missing lineups from the nearest possession in the same
    # period that has a full lineup for that team (next one first, since
    # missing data is usually at the start of a period before any subs).
    for side in ("home", "away"):
        for i, r in enumerate(rows):
            if r[side] is None:
                r[side] = _nearest_lineup(rows, i, side)
                if r[side] is not None:
                    r[f"{side}_filled"] = True

    records = []
    for r in rows:
        if r.get("home_filled") or r.get("away_filled"):
            diag["filled"] += 1
        if r["home"] is None or r["away"] is None:
            diag["skipped"] += 1
            diag["skipped_home_pts"] += r["home_pts"]
            diag["skipped_away_pts"] += r["away_pts"]
            continue
        records.append(PossessionRecord(
            period=r["period"], offense_team_id=r["offense"],
            home_players=r["home"], away_players=r["away"], seconds=r["seconds"],
            home_pts=r["home_pts"], away_pts=r["away_pts"]))
    diag["final_home"], diag["final_away"] = score
    return records, diag


def _lineup(events, team_id: int):
    """First full 5-man lineup for a team seen during the possession, else None."""
    for ev in events:
        try:
            players = (ev.current_players or {}).get(team_id, [])
        except Exception:  # noqa: BLE001 - pbpstats can raise on malformed events
            continue
        if len(players) == 5:
            return tuple(sorted(players))
    return None


def _nearest_lineup(rows: list[dict], i: int, side: str):
    period = rows[i]["period"]
    for j in range(i + 1, len(rows)):              # look ahead first
        if rows[j]["period"] != period:
            break
        if rows[j][side] is not None and not rows[j].get(f"{side}_filled"):
            return rows[j][side]
    for j in range(i - 1, -1, -1):                 # then look back
        if rows[j]["period"] != period:
            break
        if rows[j][side] is not None and not rows[j].get(f"{side}_filled"):
            return rows[j][side]
    return None


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
    # Overtime/period starters sometimes can't be read from the play-by-play
    # (a player on the floor with no recorded events). Look them up, save an
    # override, and retry. Up to 8 fixes per game (both teams, several OTs).
    for _ in range(8):
        try:
            records, diag = extract_possessions(game_id, home_team_id, away_team_id)
            break
        except InvalidNumberOfStartersException as exc:
            m = _STARTERS_ERR.search(str(exc))
            if not m:
                raise
            gid, period, team = m.group(1), int(m.group(2)), int(m.group(3))
            starters = find_period_starters(gid, period, team)
            if not starters:
                raise
            log.info("%s period %d team %d: starters from box score %s", gid, period, team, starters)
            add_starter_override(gid, period, team, starters)
    else:
        raise RuntimeError(f"{game_id}: still missing period starters after 8 fixes")
    skipped = diag["skipped"]
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
