"""Game plan for one team against an opponent lineup.

Examples:
    python scripts/plan_matchup.py --team ATL --opp NYK
    python scripts/plan_matchup.py --team ATL --opp NYK --slider 1.5
    python scripts/plan_matchup.py --team ATL --opp BOS --opp-players "Tatum" "Brown" "White" "Holiday" "Porzingis"
    python scripts/plan_matchup.py --team ATL --opp NYK --my-players "Daniels" "Johnson" "Okongwu" ...
    python scripts/plan_matchup.py --team ATL --opp NYK --size-strength 0.5

Sandbox (score your own 5 and your own assignments):
    python scripts/plan_matchup.py --team ATL --opp NYK \
        --my-players Daniels McCollum Johnson Alexander-Walker Okongwu \
        --assign Daniels=Brunson McCollum=Bridges Johnson=Anunoby Alexander-Walker=Hart Okongwu=Towns

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
    p.add_argument("--size-strength", type=float, default=1.0,
                   help="Scale the size/style matchup effects (1 = as fitted, 0 = off)")
    p.add_argument("--assign", nargs=5, default=None, metavar="MINE=THEIRS",
                   help="Sandbox: score this exact assignment (needs exactly 5 --my-players)")
    p.add_argument("--opp-players", nargs=5, default=None, help="Exactly 5 opponent players")
    p.add_argument("--my-players", nargs="+", default=None, help="Your candidate players (5+)")
    p.add_argument("--min-minutes", type=float, default=150)
    p.add_argument("--max-roster", type=int, default=13)
    p.add_argument("--top", type=int, default=10)
    return p.parse_args()


def resolve(cur, token: str, pool: list[int] | None = None) -> int:
    """Find a player by ID or name. Order of preference:
    1) players in `pool` (e.g. your roster / their lineup), 2) recent players, 3) anyone.
    Within each, an exact full-name or exact last-name match beats a partial match,
    so 'Daniels' finds Dyson Daniels, not Jaden McDaniels."""
    if token.isdigit():
        return int(token)
    t = token.strip().lower()
    cur.execute("""SELECT p.player_id, p.full_name,
                          EXISTS (SELECT 1 FROM player_team_season s WHERE s.player_id = p.player_id
                                  AND s.season = (SELECT MAX(season) FROM games)) AS recent
                   FROM players p WHERE p.full_name ILIKE %s""", (f"%{token}%",))
    hits = cur.fetchall()
    if not hits:
        raise SystemExit(f"No player matches '{token}'.")

    def quality(name: str) -> int:
        n = name.lower()
        if n == t:
            return 3
        parts = n.split()
        if parts and (parts[-1] == t or " ".join(parts[1:]) == t):
            return 2
        if any(word == t for word in parts):
            return 1
        return 0

    for scope in ([h for h in hits if pool and h[0] in pool], [h for h in hits if h[2]], hits):
        if not scope:
            continue
        best = max(quality(h[1]) for h in scope)
        top = [h for h in scope if quality(h[1]) == best]
        if len(top) == 1:
            return top[0][0]
    options = ", ".join(h[1] for h in hits[:8])
    raise SystemExit(f"'{token}' matched {len(hits)} players ({options}). Be more specific.")


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
        engine.matchup.size_strength = args.size_strength

        opp_pool = roster(cur, args.opp, season, 0, 30)
        my_pool = roster(cur, args.team, season, 0, 30)
        theirs = ([resolve(cur, t, opp_pool) for t in args.opp_players] if args.opp_players
                  else most_used_lineup(cur, args.opp, season))
        mine = ([resolve(cur, t, my_pool) for t in args.my_players] if args.my_players
                else roster(cur, args.team, season, args.min_minutes, args.max_roster))
        if len(mine) < 5:
            raise SystemExit("Need at least 5 of your players.")
        cur.execute("SELECT player_id, full_name FROM players WHERE player_id = ANY(%s)",
                    (list(set(mine) | set(theirs)),))
        name = dict(cur.fetchall())

        def nm(p):
            return name.get(p, str(p))

        print(f"\n{args.team} vs {args.opp} ({season}), defensive emphasis x{args.slider}, "
              f"size effects x{args.size_strength}")
        print(f"Opponent lineup: {', '.join(name.get(p, str(p)) for p in theirs)}")
        print(f"Searching {len(mine)} players -> every 5-man combination\n")

        if args.assign:
            if len(mine) != 5:
                raise SystemExit("--assign needs exactly 5 --my-players.")
            assignment = {}
            for pair in args.assign:
                if "=" not in pair:
                    raise SystemExit(f"Bad --assign entry '{pair}'. Use MINE=THEIRS.")
                a, b = pair.split("=", 1)
                assignment[resolve(cur, a, mine)] = resolve(cur, b, theirs)
            bad = [nm(p) for p in assignment.values() if p not in theirs]
            missing = [nm(p) for p in mine if p not in assignment]
            if bad or missing or len(set(assignment.values())) != 5:
                raise SystemExit("Each of your 5 must guard a different opponent in: "
                                 + ", ".join(nm(p) for p in theirs)
                                 + (f". Not in their lineup: {', '.join(bad)}" if bad else "")
                                 + (f". No assignment for: {', '.join(missing)}" if missing else ""))
            r = engine.evaluate(mine, theirs, assignment, args.slider)
            print("Your setup:")
            for d, s_, ppp in r["assignment"]:
                print(f"  {nm(d):<28} guards {nm(s_):<28} ({ppp:.1f} pts/100 matchup poss)")
            print(f"  net {r['net']:+.1f}  (scored {r['scored']:.1f}, allowed {r['allowed']:.1f})")
            print(f"\nBest possible with these 5: net {r['best_net']:+.1f}  "
                  f"-> your setup gives up {r['cost_vs_best']:.1f} pts/100")
            for d, s_, ppp in sorted(r["best_assignment"], key=lambda x: -x[2]):
                print(f"  {nm(d):<28} guards {nm(s_):<28} ({ppp:.1f})")
            return

        ranked = engine.rank_lineups(mine, theirs, args.slider, args.top)
        print(f"{'#':>2}  {'net':>6} {'scored':>7} {'allowed':>8}  lineup")
        for i, r in enumerate(ranked, 1):
            names = ", ".join(sorted(nm(p).split()[-1] for p in r["lineup"]))
            print(f"{i:>2}  {r['net']:+6.1f} {r['scored']:7.1f} {r['allowed']:8.1f}  {names}")

        best = ranked[0]
        options = engine.top_assignments(list(best["lineup"]), theirs, k=2)
        b = engine.cal.b
        for rank, (pairs, total) in enumerate(options, 1):
            gap = "" if rank == 1 else (f"  (+{total - options[0][1]:.1f} matchup pts/100, "
                                        f"about {b * (total - options[0][1]):+.2f} allowed per 100 after calibration)")
            print(f"\nAssignment option {rank} for the best lineup{gap}:")
            for d, s_, ppp in sorted(pairs, key=lambda x: -x[2]):
                print(f"  {nm(d):<28} guards {nm(s_):<28} ({ppp:.1f} pts/100 matchup poss)")
        print("\nnet = scored - slider x allowed, both relative to an average matchup (pts/100).")


if __name__ == "__main__":
    main()
