"""Run the full data pipeline.

Examples:
    python scripts/run_pipeline.py                          # all default seasons
    python scripts/run_pipeline.py --seasons 2025-26 2026-27
    python scripts/run_pipeline.py --seasons 2025-26 --limit 5   # quick test on 5 games
    python scripts/run_pipeline.py --steps stints            # only rebuild stints

Safe to stop and re-run at any time: finished games are skipped.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm  # noqa: E402

import config  # noqa: E402
from src import db, fetch_games, fetch_matchups, fetch_pbp_stints, fetch_teams_players  # noqa: E402
from src.utils import get_logger  # noqa: E402

log = get_logger("pipeline")
ALL_STEPS = ["meta", "stints", "matchups"]


def parse_args():
    p = argparse.ArgumentParser(description="NBA Lineup & Matchup Lab data pipeline")
    p.add_argument("--seasons", nargs="+", default=config.SEASONS)
    p.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=ALL_STEPS)
    p.add_argument("--limit", type=int, default=None,
                   help="Only process this many games per season (for testing)")
    return p.parse_args()


def season_games(conn, season: str) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute("SELECT game_id, home_team_id, away_team_id FROM games "
                    "WHERE season = %s ORDER BY game_date, game_id", (season,))
        return cur.fetchall()


def run_per_game(conn, step: str, games: list[tuple]) -> None:
    done = db.done_games(conn, step)
    todo = [g for g in games if g[0] not in done]
    log.info("%s: %d games to do (%d already done)", step, len(todo), len(games) - len(todo))
    failures = 0
    for game_id, home, away in tqdm(todo, desc=step, unit="game"):
        try:
            if step == "stints":
                n = fetch_pbp_stints.load_game(conn, game_id, home, away)
            else:
                n = fetch_matchups.load_game(conn, game_id)
            db.mark(conn, game_id, step, "done", f"{n} rows")
        except Exception as exc:  # noqa: BLE001 - log and keep going
            conn.rollback()
            failures += 1
            log.error("%s %s failed: %s", step, game_id, exc)
            db.mark(conn, game_id, step, "failed", repr(exc))
        conn.commit()  # save progress after every game
    if failures:
        log.warning("%s: %d games failed. Re-run the same command to retry them.", step, failures)


def main():
    args = parse_args()
    db.init_schema()
    with db.get_conn() as conn:
        if "meta" in args.steps:
            fetch_teams_players.load_teams(conn)
            fetch_teams_players.load_player_bios(conn, args.seasons[-1])
            for season in args.seasons:
                fetch_games.load_season(conn, season)
            conn.commit()

        for season in args.seasons:
            games = season_games(conn, season)
            if not games:
                log.warning("No games found for %s. Run with --steps meta first.", season)
                continue
            if args.limit:
                games = games[: args.limit]
            for step in ("stints", "matchups"):
                if step in args.steps:
                    log.info("=== %s: %s ===", season, step)
                    run_per_game(conn, step, games)
    log.info("Pipeline finished.")


if __name__ == "__main__":
    main()
