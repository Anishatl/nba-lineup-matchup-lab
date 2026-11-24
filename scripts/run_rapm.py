"""Fit RAPM and save player ratings.

Examples:
    python scripts/run_rapm.py                       # all seasons, recency-weighted
    python scripts/run_rapm.py --validate            # also run the out-of-sample test
    python scripts/run_rapm.py --validate-only       # just the out-of-sample test
    python scripts/run_rapm.py --decay 1.0           # no recency weighting
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src import db  # noqa: E402
from src.rapm import build_matrix, fit, save, validate  # noqa: E402

REPORT = config.ROOT / "reports" / "rapm_validation.md"


def parse_args():
    p = argparse.ArgumentParser(description="Fit RAPM player ratings")
    p.add_argument("--seasons", nargs="+", default=config.SEASONS)
    p.add_argument("--decay", type=float, default=0.5,
                   help="Weight multiplier per season back in time (1.0 = equal weights)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--min-poss", type=int, default=3000,
                   help="Min total possessions to appear in the printed top/bottom lists")
    p.add_argument("--validate", action="store_true", help="Also run the out-of-sample test")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--train", nargs="+", default=["2023-24", "2024-25"])
    p.add_argument("--test", default="2025-26")
    return p.parse_args()


def print_leaders(df, min_poss: int, n: int = 15):
    df = df[(df["off_poss"] + df["def_poss"]) >= min_poss].sort_values("total_rapm", ascending=False)
    cols = ["name", "total_rapm", "o_rapm", "d_display", "off_poss", "def_poss"]
    fmt = {"total_rapm": "{:+.2f}".format, "o_rapm": "{:+.2f}".format, "d_display": "{:+.2f}".format}
    print(f"\nTop {n} (min {min_poss:,} possessions). D shown flipped: higher = better defense")
    print(df.head(n)[cols].to_string(index=False, formatters=fmt))
    print(f"\nBottom {n}")
    print(df.tail(n)[cols].to_string(index=False, formatters=fmt))


def main():
    args = parse_args()
    db.init_schema()
    with db.get_conn() as conn:
        if not args.validate_only:
            weights = build_matrix.season_weights(args.seasons, args.decay)
            print("Season weights:", {k: round(v, 3) for k, v in sorted(weights.items())})
            stints = build_matrix.load_stints(conn, args.seasons)
            design = build_matrix.build(stints, weights)
            print(f"Design matrix: {design.X.shape[0]:,} rows x {design.X.shape[1]:,} columns "
                  f"({design.n_players:,} players)")

            print("Choosing ridge penalty with game-level cross-validation...")
            alpha, cv_table = fit.cross_validate(design, n_splits=args.folds)
            cv_rmse = float(cv_table.loc[cv_table["alpha"] == alpha, "cv_rmse"].iloc[0])
            model = fit.fit(design, alpha)
            run_id = save.save_run(conn, model, design, args.seasons, weights, cv_rmse)
            conn.commit()

            print(f"\nRun {run_id}: alpha={alpha:,.0f}  intercept={model.intercept:.2f} pts/100  "
                  f"home advantage={model.home_adv:+.2f} pts/100")
            print_leaders(save.named_ratings(conn, model.ratings_frame(design)), args.min_poss)

        if args.validate or args.validate_only:
            print()
            result = validate.run(conn, args.train, args.test, args.decay, n_splits=args.folds)
            print()
            print(result.table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            REPORT.parent.mkdir(parents=True, exist_ok=True)
            REPORT.write_text(result.to_markdown(), encoding="utf-8")
            print(f"\nSaved {REPORT.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
