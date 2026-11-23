"""Sanity-check stint points against official final scores.

Re-parses already-downloaded games (from the cache, no new downloads) and shows
where any missing points went.

    python scripts/check_stints.py              # games already loaded (max 20)
    python scripts/check_stints.py --limit 50
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import db, fetch_pbp_stints  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args()

    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT g.game_id, g.home_team_id, g.away_team_id, g.home_pts, g.away_pts
            FROM games g JOIN pipeline_log l ON l.game_id = g.game_id
            WHERE l.step = 'stints' AND l.status = 'done'
            ORDER BY g.game_id LIMIT %s""", (args.limit,))
        games = cur.fetchall()

    if not games:
        print("No loaded games yet. Run the pipeline first.")
        return

    header = (f"{'game':<12}{'official':>10}{'feed':>10}{'in stints':>11}"
              f"{'skipped':>9}{'skip pts':>10}{'counted':>10}")
    print(header)
    print("-" * len(header))
    for game_id, home, away, off_h, off_a in games:
        records, d = fetch_pbp_stints.extract_possessions(game_id, home, away)
        in_h = sum(r.home_pts for r in records)
        in_a = sum(r.away_pts for r in records)
        cols = [
            f"{off_h}-{off_a}",
            "{}-{}".format(d["final_home"], d["final_away"]),
            f"{in_h}-{in_a}",
            "{}/{}".format(d["skipped"], d["possessions"]),
            "{}-{}".format(d["skipped_home_pts"], d["skipped_away_pts"]),
            "{}-{}".format(d["counted_home_pts"], d["counted_away_pts"]),
        ]
        print(f"{game_id:<12}{cols[0]:>10}{cols[1]:>10}{cols[2]:>11}"
              f"{cols[3]:>9}{cols[4]:>10}{cols[5]:>10}")
    print("\nofficial = box score | feed = final score in play-by-play | "
          "in stints = points kept\nskipped = possessions dropped for incomplete "
          "lineups | counted = old make-counting method")


if __name__ == "__main__":
    main()
