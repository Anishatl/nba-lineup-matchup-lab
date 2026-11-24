"""Game plan for one team against an opponent lineup.

Examples:
    python scripts/plan_matchup.py --team ATL --opp NYK
    python scripts/plan_matchup.py --team ATL --opp NYK --slider 1.5
    python scripts/plan_matchup.py --team ATL --opp BOS --opp-players "Tatum" "Brown" "White" "Holiday" "Porzingis"
    python scripts/plan_matchup.py --team ATL --opp NYK --my-players "Daniels" "Johnson" "Okongwu" ...

Players can be given by (part of) their name or by NBA player ID, so any
players can be used on either side.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import db  # noqa: E402
from src.matchups import store  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Best lineups and defensive assignments vs an opponent")
    p.add_argument("--team", required=True, help="Your team abbreviation, e.g. ATL")
    p.add_argument("--opp", required=True, help="Opponent abbreviation, e.g. NYK")
    p.add_argument("--season", default=None, help="Season for rosters (default: latest)")
    p.add_argument("--slider", type=float, default=1.0, help="Defensive emphasis (1.0 = plain net rating)")
    p.add_argument("--opp-players", nargs=5, default=None, help="Exactly 5 opponent players")
    p.add_argument("--my-players", nargs="+", default=None, help="Your candidate players (5+)")
    p.add_argument("--min-minutes", type=float, default=150)
    p.add_argument("--max-roster", type=int, default=13)
    p.add_argument("--top", type=int, default=10)
    return p.parse_args()


def resolve(cur, token: str) -> int:
    if token.isdigit():
        return int(token)
    cur.execute("SELECT player_id, full_name FROM players WHERE full_name ILIKE %s", (f"%{token}%",))
    hits = cur.fetchall()
    if len(hits) == 1:
        return hits[0][0]
    # prefer players who actually appear in recent data
    cur.execute("""SELECT DISTINCT p.player_id, p.full_name FROM players p
                   JOIN player_team_season s USING (player_id)
                   WHERE p.full_name ILIKE %s AND s.season = (SELECT MAX(season) FROM games)""",
                (f"%{token}%",))
    recent = cur.fetchall()
    if len(recent) == 1:
        return recent[0][0]
    options = ", ".join(n for _, n in (recent or hits)[:8]) or "none"
    raise SystemExit(f"'{token}' matched {len(recent or hits)} players ({options}). Be more specific.")


def roster(cur, team, season, min_minutes, max_n) -> list[int]:
    cur.execute("""SELECT s.player_id FROM player_team_season s JOIN teams t USING (team_id)
                   WHERE t.abbreviation = %s AND s.season = %s AND s.minutes >= %s
                   ORDER BY s.minutes DESC LIMIT %s""", (team, season, min_minutes, max_n))
    return [r[0] for r in cur.fetchall()]


def most_used_lineup(cur, team, season) -> list[int]:
    cur.execute("""
        WITH t AS (SELECT team_id FROM teams WHERE abbreviation = %s),
        sides AS (
            SELECT CASE WHEN g.home_team_id = t.team_id THEN s.home_players ELSE s.away_players END AS lineup,
                   s.home_poss + s.away_poss AS poss
            FROM stints s JOIN games g USING (game_id) CROSS JOIN t
            WHERE g.season = %s AND t.team_id IN (g.home_team_id, g.away_team_id))
        SELECT lineup FROM sides GROUP BY lineup ORDER BY SUM(poss) DESC LIMIT 1""", (team, season))
    row = cur.fetchone()
    if not row:
        raise SystemExit(f"No lineups found for {team} in {season}.")
    return list(row[0])


def main():
    args = parse_args()
    with db.get_conn() as conn, conn.cursor() as cur:
        season = args.season
        if season is None:
            cur.execute("SELECT MAX(season) FROM games")
            season = cur.fetchone()[0]
        engine = store.load_engine(conn)

        theirs = ([resolve(cur, t) for t in args.opp_players] if args.opp_players
                  else most_used_lineup(cur, args.opp, season))
        mine = ([resolve(cur, t) for t in args.my_players] if args.my_players
                else roster(cur, args.team, season, args.min_minutes, args.max_roster))
        if len(mine) < 5:
            raise SystemExit("Need at least 5 of your players.")
        cur.execute("SELECT player_id, full_name FROM players WHERE player_id = ANY(%s)",
                    (list(set(mine) | set(theirs)),))
        name = dict(cur.fetchall())

        print(f"\n{args.team} vs {args.opp} ({season}), defensive emphasis x{args.slider}")
        print(f"Opponent lineup: {', '.join(name.get(p, str(p)) for p in theirs)}")
        print(f"Searching {len(mine)} players -> every 5-man combination\n")

        ranked = engine.rank_lineups(mine, theirs, args.slider, args.top)
        print(f"{'#':>2}  {'net':>6} {'scored':>7} {'allowed':>8}  lineup")
        for i, r in enumerate(ranked, 1):
            names = ", ".join(sorted(name.get(p, str(p)).split()[-1] for p in r["lineup"]))
            print(f"{i:>2}  {r['net']:+6.1f} {r['scored']:7.1f} {r['allowed']:8.1f}  {names}")

        best = ranked[0]
        print("\nBest lineup's defensive assignments (Hungarian algorithm):")
        for d, s, ppp in sorted(best["assignment"], key=lambda x: -x[2]):
            print(f"  {name.get(d, d):<28} guards {name.get(s, s):<28} ({ppp:.1f} pts/100 matchup poss)")
        print("\nnet = scored - slider x allowed, both relative to an average matchup (pts/100).")


if __name__ == "__main__":
    main()
