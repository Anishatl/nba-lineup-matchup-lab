"""Fit the matchup model, validate it, calibrate it against RAPM, and save.

Run scripts/run_rapm.py first (calibration uses the latest RAPM run's penalty).

Examples:
    python scripts/run_matchups.py                  # Poisson model: fit + validate + calibrate + save
    python scripts/run_matchups.py --kind linear    # additive model instead
    python scripts/run_matchups.py --skip-validate  # faster: no model comparison
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from src import db  # noqa: E402
from src.matchups import calibrate, model as mm, store, validate  # noqa: E402
from src.rapm.build_matrix import season_weights  # noqa: E402

REPORT = config.ROOT / "reports" / "matchup_validation.md"


def parse_args():
    p = argparse.ArgumentParser(description="Fit + calibrate the matchup model")
    p.add_argument("--kind", choices=["poisson", "linear"], default="poisson")
    p.add_argument("--seasons", nargs="+", default=config.SEASONS)
    p.add_argument("--decay", type=float, default=0.5)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--train", nargs="+", default=["2023-24", "2024-25"])
    p.add_argument("--test", default="2025-26")
    p.add_argument("--skip-validate", action="store_true")
    p.add_argument("--no-context", action="store_true",
                   help="Fit without team-season context terms")
    p.add_argument("--min-poss", type=float, default=1500,
                   help="Min matchup possessions to appear in the printed lists")
    return p.parse_args()


def print_leaders(conn, model: mm.MatchupModel, min_poss: float, n: int = 15):
    eff = model.total_effects()
    eff["def_poss"], eff["off_poss"] = model.poss_def, model.poss_off
    with conn.cursor() as cur:
        cur.execute("SELECT player_id, full_name FROM players WHERE player_id = ANY(%s)",
                    ([int(x) for x in eff.player_id],))
        eff["name"] = eff.player_id.map(dict(cur.fetchall()))
    fmt = {"defender_pct": "{:+.1f}%".format, "scorer_pct": "{:+.1f}%".format,
           "def_poss": "{:,.0f}".format, "off_poss": "{:,.0f}".format}
    d = eff[eff.def_poss >= min_poss].sort_values("defender_pct")
    s = eff[eff.off_poss >= min_poss].sort_values("scorer_pct", ascending=False)
    print(f"\nBest one-on-one defenders (min {min_poss:,.0f} matchup poss).")
    print("defender_pct = change in his man's scoring vs. an average defender, size terms included")
    print(d.head(n)[["name", "defender_pct", "def_poss"]].to_string(index=False, formatters=fmt))
    print(f"\nToughest scorers to guard (min {min_poss:,.0f} matchup poss)")
    print(s.head(n)[["name", "scorer_pct", "off_poss"]].to_string(index=False, formatters=fmt))


def print_team_context(conn, model: mm.MatchupModel, n: int = 8):
    ctx = model.team_context()
    if ctx.empty:
        return
    with conn.cursor() as cur:
        cur.execute("SELECT team_id, abbreviation FROM teams")
        ctx["team"] = ctx.team_id.map(dict(cur.fetchall()))
    unit = "%" if model.kind == "poisson" else " pts/100"
    fmt = {"effect": lambda x: f"{x:+.1f}{unit}"}
    d = ctx[ctx.side == "defense"].sort_values("effect")
    print("\nTeam defensive context (scheme/help effect on every matchup; negative = suppresses scoring)")
    print(d.head(n)[["team", "season", "effect"]].to_string(index=False, formatters=fmt))
    print("...")
    print(d.tail(3)[["team", "season", "effect"]].to_string(index=False, header=False, formatters=fmt))


def main():
    args = parse_args()
    db.init_schema()
    with db.get_conn() as conn:
        weights = season_weights(args.seasons, args.decay)
        rows = mm.load_matchups(conn, args.seasons)
        feats, meta = mm.player_features(conn, rows)
        design = mm.build(mm.aggregate(rows), feats, meta, weights)
        print(f"Matchup rows: {len(rows):,} ({len(design.rate):,} after aggregating)  "
              f"players: {design.n_players:,}  model: {args.kind}")
        print("Choosing ridge penalty with game-level cross-validation...")
        which = "full" if args.no_context else "full+context"
        alpha, cv = mm.cross_validate(rows, feats, meta, weights, args.kind, n_splits=args.folds, which=which)
        cv_score = float(cv.loc[cv.alpha == alpha].iloc[0, 1])
        model = mm.fit(design, feats, meta, alpha, args.kind, which)
        if args.kind == "poisson":
            print(f"alpha={alpha:.3g}  league rate={100 * np.exp(model.intercept):.2f} pts/100 matchup poss")
            print("size/style effects:", {n: f"{100 * (np.exp(w) - 1):+.2f}% per unit"
                                          for n, w in zip(mm.FEATURES, model.feat_coef)})
        else:
            print(f"alpha={alpha:,.0f}  intercept={model.intercept:.2f} pts/100 matchup poss")
            print("size/style weights:", {n: round(float(w), 3) for n, w in zip(mm.FEATURES, model.feat_coef)})

        report = []
        val_alpha = alpha
        if not args.skip_validate:
            print()
            v = validate.run(conn, args.train, args.test, args.decay, n_splits=args.folds)
            val_alpha = v.alphas["poisson+context" if args.kind == "poisson" and not args.no_context
                                 else args.kind]
            print()
            print(v.table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            report.append(v.to_markdown())

        with conn.cursor() as cur:
            cur.execute("SELECT alpha FROM model_runs WHERE model = 'rapm' ORDER BY run_id DESC LIMIT 1")
            row = cur.fetchone()
        if row is None:
            raise SystemExit("No RAPM run found. Run scripts/run_rapm.py first.")
        if args.skip_validate:  # need a penalty fit on the training seasons only
            tr_rows = mm.load_matchups(conn, args.train)
            tr_feats, tr_meta = mm.player_features(conn, tr_rows)
            val_alpha, _ = mm.cross_validate(tr_rows, tr_feats, tr_meta, season_weights(args.train, args.decay),
                                             args.kind, n_splits=args.folds, which=which)
        print()
        cal = calibrate.run(conn, args.train, args.test, args.decay,
                            rapm_alpha=float(row[0]), matchup_alpha=val_alpha, kind=args.kind)
        c = cal.cal
        print(f"  base={c.base:.2f}  a={c.a:.3f}  b={c.b:.3f}  c={c.c:.3f}  home={cal.home:+.2f}")
        print(cal.table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        report.append(cal.to_markdown())

        run_id = store.save(conn, model, args.seasons, weights, cv_score, len(rows),
                            calibration=c, home=cal.home, notes=args.kind)
        conn.commit()
        print(f"\nSaved matchup run {run_id} ({args.kind})")
        print_leaders(conn, model, args.min_poss)
        print_team_context(conn, model)

        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text("\n".join(report), encoding="utf-8")
        print(f"\nSaved {REPORT.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
