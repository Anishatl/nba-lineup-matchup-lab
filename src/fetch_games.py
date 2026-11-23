"""Load every game + who played for which team, two calls per season."""
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

import config
from src import db
from src.utils import get_logger, with_retries

log = get_logger("games")


def _game_log(season: str, kind: str) -> pd.DataFrame:
    resp = with_retries(leaguegamelog.LeagueGameLog, season=season,
                        season_type_all_star=config.SEASON_TYPE,
                        player_or_team_abbreviation=kind,
                        timeout=config.REQUEST_TIMEOUT)
    return resp.get_data_frames()[0]


def games_from_team_log(df: pd.DataFrame, season: str) -> list[tuple]:
    """Each game appears twice (once per team). MATCHUP is 'ATL vs. BOS' for
    the home team and 'ATL @ BOS' for the away team."""
    games = {}
    for r in df.itertuples(index=False):
        g = games.setdefault(r.GAME_ID, {"date": r.GAME_DATE})
        side = "home" if " vs. " in r.MATCHUP else "away"
        g[f"{side}_team"] = int(r.TEAM_ID)
        g[f"{side}_pts"] = int(r.PTS) if pd.notna(r.PTS) else None
    rows = []
    for gid, g in games.items():
        if "home_team" in g and "away_team" in g:
            rows.append((gid, season, g["date"], g["home_team"], g["away_team"],
                         g.get("home_pts"), g.get("away_pts")))
    return rows


def load_season(conn, season: str) -> list[str]:
    """Load games and player_team_season for a season. Returns game IDs."""
    team_df = _game_log(season, "T")
    rows = games_from_team_log(team_df, season)
    db.upsert(conn, "games",
              ["game_id", "season", "game_date", "home_team_id", "away_team_id",
               "home_pts", "away_pts"],
              rows, conflict=["game_id"], update=["home_pts", "away_pts"])
    log.info("%s: %d games", season, len(rows))

    player_df = _game_log(season, "P")
    # Make sure every player exists (bios may be missing for brand-new players)
    names = player_df[["PLAYER_ID", "PLAYER_NAME"]].drop_duplicates("PLAYER_ID")
    db.upsert(conn, "players", ["player_id", "full_name"],
              [(int(p), n) for p, n in names.itertuples(index=False)],
              conflict=["player_id"])
    pts = (player_df.groupby(["PLAYER_ID", "TEAM_ID"])
           .agg(games=("GAME_ID", "nunique"), minutes=("MIN", "sum")).reset_index())
    db.upsert(conn, "player_team_season",
              ["player_id", "team_id", "season", "games", "minutes"],
              [(int(r.PLAYER_ID), int(r.TEAM_ID), season, int(r.games), float(r.minutes or 0))
               for r in pts.itertuples(index=False)],
              conflict=["player_id", "team_id", "season"], update=["games", "minutes"])
    log.info("%s: %d player-team rows", season, len(pts))
    return [r[0] for r in rows]
