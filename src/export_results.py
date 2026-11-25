"""Export the latest model results as small JSON files for the static
dashboard (docs/data/, served by GitHub Pages).

    python -m src.export_results          (or: python src/export_results.py)

Writes:
    meta.json      model constants the browser needs (matchup model, calibration),
                   seasons, run ids, last-updated date
    players.json   every player: team, size, RAPM (raw + display), matchup skills,
                   % effects, calibrated defensive impact, flags
    teams.json     every team: roster this season (by minutes), most-used lineups,
                   defensive context
    fixtures.json  test cases computed by the Python engine; the site's JS model is
                   checked against them (docs/test.html, scripts/check_js.mjs)

Display convention: O-RAPM and D-RAPM are re-centered so 0 = the possession-
weighted average player, and D is flipped so higher = better defense. The raw
coefficients (used for all math) are exported too.
"""
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from src import db  # noqa: E402
from src.matchups import model as mm  # noqa: E402
from src.matchups import store  # noqa: E402

OUT = config.SITE_DATA_DIR


def _r(x, n=4):
    """Round for JSON; None for missing."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), n)


def _latest(cur, model):
    cur.execute("SELECT MAX(run_id) FROM model_runs WHERE model = %s", (model,))
    return cur.fetchone()[0]


def _write(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"  {path}  ({path.stat().st_size / 1024:,.0f} KB)")


def export_meta(cur, engine, rapm_run, m_run, season):
    m = engine.matchup
    cur.execute("SELECT seasons, intercept, home_adv, created_at FROM model_runs WHERE run_id = %s", (rapm_run,))
    seasons, r_int, r_home, r_created = cur.fetchone()
    cur.execute("SELECT name, value FROM model_params WHERE run_id = %s", (m_run,))
    p = {k: float(v) for k, v in cur.fetchall()}
    return {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "season": season,
        "seasons": list(seasons),
        "rapm": {"run_id": rapm_run, "intercept": _r(r_int), "home": _r(r_home)},
        "matchup": {
            "run_id": m_run, "kind": m.kind, "intercept": _r(m.intercept, 6),
            "features": mm.FEATURES, "weights": [_r(w, 6) for w in m.feat_coef],
            "height_mean": _r(m.meta["height_mean"], 6),
            "three_rate_mean": _r(m.meta["three_rate_mean"], 6),
            "centers": [_r(m.meta.get(f"center_{f}", 0.0), 6) for f in mm.FEATURES],
        },
        "calibration": {"base": _r(engine.cal.base, 6), "a": _r(engine.cal.a, 6), "b": _r(engine.cal.b, 6),
                        "c": _r(engine.cal.c, 6), "dm_mean": _r(engine.cal.dm_mean, 6),
                        "home": _r(p.get("cal_home"))},
    }


def export_coverage(cur):
    """Games per season, stints loaded, exact box-score matches, matchup coverage."""
    cur.execute("""
        WITH sp AS (SELECT game_id, SUM(home_pts) hp, SUM(away_pts) ap FROM stints GROUP BY game_id),
             mg AS (SELECT DISTINCT game_id FROM matchups)
        SELECT g.season, COUNT(*) AS games, COUNT(sp.game_id) AS stints,
               SUM(CASE WHEN sp.hp = g.home_pts AND sp.ap = g.away_pts THEN 1 ELSE 0 END) AS exact,
               COUNT(mg.game_id) AS matchups
        FROM games g LEFT JOIN sp USING (game_id) LEFT JOIN mg USING (game_id)
        GROUP BY g.season ORDER BY g.season""")
    rows = [{"season": s, "games": int(g), "stints": int(st), "exact": int(e or 0), "matchups": int(m)}
            for s, g, st, e, m in cur.fetchall()]
    cur.execute("SELECT COUNT(*) FROM stints")
    n_stints = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM matchups")
    n_matchups = cur.fetchone()[0]
    return {"seasons": rows, "stint_rows": int(n_stints), "matchup_rows": int(n_matchups)}


def export_validation():
    out = {}
    for key, name in (("rapm", "rapm_validation.json"), ("matchup", "matchup_validation.json")):
        path = config.ROOT / "reports" / name
        if path.exists():
            out[key] = json.loads(path.read_text(encoding="utf-8"))
    return out


def export_players(cur, rapm_run, m_run, season):
    cur.execute("""
        SELECT p.player_id, p.full_name, p.position, r.o_rapm, r.d_rapm, r.off_poss, r.def_poss
        FROM players p JOIN player_ratings r USING (player_id) WHERE r.run_id = %s""", (rapm_run,))
    rp = pd.DataFrame(cur.fetchall(), columns=["id", "name", "pos", "o", "d", "o_poss", "d_poss"])
    cur.execute("""
        SELECT player_id, scorer_skill, defender_skill, off_poss, def_poss, height_inches, weight_lbs,
               three_rate, defender_pct, scorer_pct, pts_saved, def_flag
        FROM matchup_ratings WHERE run_id = %s""", (m_run,))
    mr = pd.DataFrame(cur.fetchall(), columns=["id", "sc", "df", "m_off", "m_def", "h", "w", "t",
                                               "df_pct", "sc_pct", "saved", "flag"])
    df = rp.merge(mr, on="id", how="outer")
    if df["name"].isna().any():
        cur.execute("SELECT player_id, full_name, position FROM players WHERE player_id = ANY(%s)",
                    ([int(i) for i in df.loc[df["name"].isna(), "id"]],))
        fill = {i: (n, pos) for i, n, pos in cur.fetchall()}
        for i in df.index[df["name"].isna()]:
            df.at[i, "name"], df.at[i, "pos"] = fill.get(int(df.at[i, "id"]), (str(df.at[i, "id"]), None))
    num = ["o", "d", "o_poss", "d_poss", "sc", "df", "m_off", "m_def", "h", "w", "t", "df_pct", "sc_pct", "saved"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")

    # Display RAPM: 0 = possession-weighted average player, D flipped (higher = better)
    o_mean = np.average(df["o"].fillna(0), weights=df["o_poss"].fillna(0) + 1e-9)
    d_mean = np.average(df["d"].fillna(0), weights=df["d_poss"].fillna(0) + 1e-9)
    df["o_disp"] = df["o"] - o_mean
    df["d_disp"] = -(df["d"] - d_mean)

    # Current team = most minutes in the latest season; else latest season played
    cur.execute("""
        SELECT DISTINCT ON (s.player_id) s.player_id, t.abbreviation, s.season, s.minutes
        FROM player_team_season s JOIN teams t USING (team_id)
        ORDER BY s.player_id, s.season DESC, s.minutes DESC""")
    team = {pid: (abbr, s) for pid, abbr, s, _ in cur.fetchall()}

    out = []
    for r in df.itertuples(index=False):
        abbr, last = team.get(int(r.id), (None, None))
        out.append({
            "id": int(r.id), "name": r.name, "pos": r.pos, "team": abbr, "active": last == season,
            "h": _r(r.h, 1), "t": _r(r.t, 4),
            "o": _r(r.o, 4), "d": _r(r.d, 4),
            "o_disp": _r(r.o_disp, 2), "d_disp": _r(r.d_disp, 2),
            "tot_disp": _r((r.o_disp if pd.notna(r.o_disp) else 0) + (r.d_disp if pd.notna(r.d_disp) else 0), 2),
            "poss": int((r.o_poss if pd.notna(r.o_poss) else 0) + (r.d_poss if pd.notna(r.d_poss) else 0)),
            "sc": _r(r.sc, 5), "df": _r(r.df, 5),
            "sc_pct": _r(r.sc_pct, 1), "df_pct": _r(r.df_pct, 1),
            "m_off": int(r.m_off) if pd.notna(r.m_off) else 0,
            "m_def": int(r.m_def) if pd.notna(r.m_def) else 0,
            "saved": _r(r.saved, 2), "flag": r.flag or "",
        })
    return out, {"o_mean": _r(o_mean, 4), "d_mean": _r(d_mean, 4)}


def export_teams(cur, season, m_run):
    cur.execute("SELECT team_id, abbreviation, city, name FROM teams ORDER BY abbreviation")
    teams = {}
    for tid, abbr, city, name in cur.fetchall():
        cur.execute("""SELECT player_id, minutes, games FROM player_team_season
                       WHERE team_id = %s AND season = %s ORDER BY minutes DESC""", (tid, season))
        roster = [{"id": p, "min": round(float(m or 0)), "g": int(g or 0)} for p, m, g in cur.fetchall()]
        cur.execute("""
            WITH sides AS (
                SELECT CASE WHEN g.home_team_id = %(t)s THEN s.home_players ELSE s.away_players END AS lineup,
                       CASE WHEN g.home_team_id = %(t)s THEN s.home_poss ELSE s.away_poss END AS op,
                       CASE WHEN g.home_team_id = %(t)s THEN s.away_poss ELSE s.home_poss END AS dp,
                       CASE WHEN g.home_team_id = %(t)s THEN s.home_pts ELSE s.away_pts END AS pf,
                       CASE WHEN g.home_team_id = %(t)s THEN s.away_pts ELSE s.home_pts END AS pa,
                       s.seconds
                FROM stints s JOIN games g USING (game_id)
                WHERE g.season = %(s)s AND %(t)s IN (g.home_team_id, g.away_team_id))
            SELECT lineup, SUM(op), SUM(dp), SUM(pf), SUM(pa), SUM(seconds)
            FROM sides GROUP BY lineup ORDER BY SUM(op) + SUM(dp) DESC LIMIT 15""", {"t": tid, "s": season})
        lineups = []
        for l, op, dp, pf, pa, sec in cur.fetchall():
            op, dp, pf, pa = int(op or 0), int(dp or 0), int(pf or 0), int(pa or 0)
            net = (100.0 * pf / op if op else 0) - (100.0 * pa / dp if dp else 0)
            lineups.append({"ids": list(l), "poss": op + dp, "op": op, "dp": dp, "pf": pf, "pa": pa,
                            "min": round(float(sec or 0) / 60, 1), "net": _r(net, 1)})
        # season record and team totals
        cur.execute("""
            SELECT COUNT(*) FILTER (WHERE (home_team_id = %(t)s AND home_pts > away_pts)
                                       OR (away_team_id = %(t)s AND away_pts > home_pts)),
                   COUNT(*) FILTER (WHERE (home_team_id = %(t)s AND home_pts < away_pts)
                                       OR (away_team_id = %(t)s AND away_pts < home_pts)),
                   SUM(CASE WHEN home_team_id = %(t)s THEN home_pts ELSE away_pts END),
                   SUM(CASE WHEN home_team_id = %(t)s THEN away_pts ELSE home_pts END)
            FROM games WHERE season = %(s)s AND %(t)s IN (home_team_id, away_team_id)""", {"t": tid, "s": season})
        w, l_, pf_tot, pa_tot = cur.fetchone()
        cur.execute("""
            SELECT SUM(CASE WHEN g.home_team_id = %(t)s THEN s.home_pts ELSE s.away_pts END),
                   SUM(CASE WHEN g.home_team_id = %(t)s THEN s.home_poss ELSE s.away_poss END),
                   SUM(CASE WHEN g.home_team_id = %(t)s THEN s.away_pts ELSE s.home_pts END),
                   SUM(CASE WHEN g.home_team_id = %(t)s THEN s.away_poss ELSE s.home_poss END)
            FROM stints s JOIN games g USING (game_id)
            WHERE g.season = %(s)s AND %(t)s IN (g.home_team_id, g.away_team_id)""", {"t": tid, "s": season})
        tpf, top, tpa, tdp = [float(x or 0) for x in cur.fetchone()]
        record = {"w": int(w or 0), "l": int(l_ or 0), "pf": int(pf_tot or 0), "pa": int(pa_tot or 0),
                  "ortg": _r(100 * tpf / top, 1) if top else None, "drtg": _r(100 * tpa / tdp, 1) if tdp else None}
        cur.execute("SELECT value FROM model_params WHERE run_id = %s AND name = %s",
                    (m_run, f"ctx_d|{tid}|{season}"))
        row = cur.fetchone()
        ctx = _r(100 * (np.exp(float(row[0])) - 1), 1) if row else None
        teams[abbr] = {"id": tid, "city": city, "name": name, "roster": roster,
                       "lineups": lineups, "record": record, "def_context_pct": ctx}
    return teams


def export_fixtures(engine, teams):
    """Cases computed by the Python engine, for checking the JS model."""
    rng = np.random.default_rng(0)
    ranked = sorted(teams.items(), key=lambda kv: -sum(r["min"] for r in kv[1]["roster"]))
    (a_abbr, a), (b_abbr, b) = ranked[0], ranked[1]
    mine = [r["id"] for r in a["roster"][:9]]
    theirs = b["lineups"][0]["ids"] if b["lineups"] else [r["id"] for r in b["roster"][:5]]
    ids = engine.matchup.player_ids
    pairs = [[int(x), int(y)] for x, y in rng.choice(ids, size=(12, 2))] + [[-1, int(ids[0])]]

    fx = {"team": a_abbr, "opp": b_abbr, "mine": mine, "theirs": theirs, "pairs": [], "rank": [], "evaluate": None}
    for size in (1.0, 0.5):
        engine.matchup.size_strength = size
        preds = engine.matchup.predict_pairs([p[0] for p in pairs], [p[1] for p in pairs])
        fx["pairs"].append({"size": size, "pairs": pairs, "pred": [_r(x, 6) for x in preds]})
    for slider, size in ((1.0, 1.0), (1.5, 1.0), (1.0, 0.0)):
        engine.matchup.size_strength = size
        top = engine.rank_lineups(mine, theirs, slider, top=3)
        fx["rank"].append({"slider": slider, "size": size, "top": [
            {"lineup": sorted(int(x) for x in t["lineup"]), "net": _r(t["net"], 6),
             "scored": _r(t["scored"], 6), "allowed": _r(t["allowed"], 6)} for t in top]})
    engine.matchup.size_strength = 1.0
    best = engine.rank_lineups(mine, theirs, 1.0, top=1)[0]
    five = list(best["lineup"])
    assignment = {five[i]: theirs[(i + 1) % 5] for i in range(5)}     # deliberately shuffled
    ev = engine.evaluate(five, theirs, assignment, 1.0)
    fx["evaluate"] = {"mine": [int(x) for x in five], "assignment": {str(k): int(v) for k, v in assignment.items()},
                      "net": _r(ev["net"], 6), "best_net": _r(ev["best_net"], 6),
                      "cost": _r(ev["cost_vs_best"], 6)}
    return fx


def main():
    with db.get_conn() as conn, conn.cursor() as cur:
        rapm_run, m_run = _latest(cur, "rapm"), _latest(cur, "matchup")
        if rapm_run is None or m_run is None:
            raise SystemExit("Run scripts/run_rapm.py and scripts/run_matchups.py first.")
        cur.execute("SELECT MAX(season) FROM games")
        season = cur.fetchone()[0]
        engine = store.load_engine(conn)
        print(f"Exporting RAPM run {rapm_run}, matchup run {m_run}, season {season}")
        meta = export_meta(cur, engine, rapm_run, m_run, season)
        players, centering = export_players(cur, rapm_run, m_run, season)
        meta["rapm"].update(centering)
        meta["coverage"] = export_coverage(cur)
        meta["validation"] = export_validation()
        if not meta["validation"]:
            print("  note: no reports/*.json yet; run run_rapm.py --validate and run_matchups.py "
                  "to show validation results on the Methodology page")
        teams = export_teams(cur, season, m_run)
        _write("meta.json", meta)
        _write("players.json", players)
        _write("teams.json", teams)
        _write("fixtures.json", export_fixtures(engine, teams))


if __name__ == "__main__":
    main()
