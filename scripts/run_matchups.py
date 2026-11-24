"""Fit the matchup model, validate it, calibrate it against RAPM, and save.

Run scripts/run_rapm.py first (calibration uses the latest RAPM run's penalty).

Examples:
    python scripts/run_matchups.py                  # fit + validate + calibrate + save
    python scripts/run_matchups.py --skip-validate  # fit + calibrate + save (faster)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src import db  # noqa: E402
from src.matchups import calibrate, model as mm, store, validate  # noqa: E402
from src.rapm.build_matrix import season_weights  # noqa: E402

REPORT = config.ROOT / "reports" / "matchup_validation.md"


def parse_args():
    p = argparse.ArgumentParser(description="Fit + calibrate the matchup model")
    p.add_argument("--seasons", nargs="+", default=config.SEASONS)
    p.add_argument("--decay", type=float, default=0.5)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--train", nargs="+", default=["2023-24", "2024-25"])
    p.add_argument("--test", default="2025-26")
    p.add_argument("--skip-validate", action="store_true")
    p.add_argument("--min-poss", type=float, default=1500,
                   help="Min matchup possessions to appear in the printed lists")
    return p.parse_args()


def print_leaders(conn, model: mm.MatchupModel, min_poss: float, n: int = 15):
    import pandas as pd
    df = pd.DataFrame({"player_id": model.player_ids, "scorer": model.scorer,
                       "defender": model.defender, "off_poss": model.poss_off, "def_poss": model.poss_def})
    with conn.cursor() as cur:
        cur.execute("SELECT player_id, full_name FROM players WHERE player_id = ANY(%s)",
                    ([int(x) for x in df.player_id],))
        df["name"] = df.player_id.map(dict(cur.fetchall()))
    d = df[df.def_poss >= min_poss].sort_values("defender")
    s = df[df.off_poss >= min_poss].sort_values("scorer", ascending=False)
    print(f"\nBest one-on-one defenders (min {min_poss:,.0f} matchup poss; negative = fewer pts allowed)")
    print(d.head(n)[["name", "defender", "def_poss"]].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print(f"\nToughest scorers to guard (min {min_poss:,.0f} matchup poss)")
    print(s.head(n)[["name", "scorer", "off_poss"]].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))


def main():
    args = parse_args()
    db.init_schema()
    with db.get_conn() as conn:
        weights = season_weights(args.seasons, args.decay)
        rows = mm.load_matchups(conn, args.seasons)
        feats, meta = mm.player_features(conn, rows)
        design = mm.build(rows, feats, meta, weights)
        print(f"Matchup rows: {len(rows):,}  players: {design.n_players:,}")
        print("Choosing ridge penalty with game-level cross-validation...")
        alpha, cv = mm.cross_validate(design, n_splits=args.folds)
        cv_rmse = float(cv.loc[cv.alpha == alpha, "cv_rmse"].iloc[0])
        model = mm.fit(design, feats, meta, alpha)
        print(f"alpha={alpha:,.0f}  intercept={model.intercept:.2f} pts/100 matchup poss")
        print("size/style weights:", {n: round(float(w), 3) for n, w in zip(mm.FEATURES, model.feat_coef)})

        report = []
        val_alpha = alpha
        if not args.skip_validate:
            print()
            v = validate.run(conn, args.train, args.test, args.decay, n_splits=args.folds)
            val_alpha = v.alpha
            print()
            print(v.table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            report.append(v.to_markdown())

        with conn.cursor() as cur:
            cur.execute("SELECT alpha FROM model_runs WHERE model = 'rapm' ORDER BY run_id DESC LIMIT 1")
            row = cur.fetchone()
        if row is None:
            raise SystemExit("No RAPM run found. Run scripts/run_rapm.py first.")
        print()
        cal = calibrate.run(conn, args.train, args.test, args.decay,
                            rapm_alpha=float(row[0]), matchup_alpha=val_alpha)
        c = cal.cal
        print(f"  base={c.base:.2f}  a={c.a:.3f}  b={c.b:.3f}  c={c.c:.3f}  home={cal.home:+.2f}")
        print(cal.table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        report.append(cal.to_markdown())

        run_id = store.save(conn, model, args.seasons, weights, cv_rmse, len(rows),
                            calibration=c, home=cal.home)
        conn.commit()
        print(f"\nSaved matchup run {run_id}")
        print_leaders(conn, model, args.min_poss)

        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text("\n".join(report), encoding="utf-8")
        print(f"\nSaved {REPORT.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
