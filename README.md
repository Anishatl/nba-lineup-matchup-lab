# NBA Lineup & Matchup Lab

Pick any NBA team and find out:

1. **Which 5-man lineups are actually best.** Every player's offensive and defensive impact is estimated with **RAPM** (regularized adjusted plus-minus, a ridge regression over play-by-play stints), then every possible lineup on the roster is ranked.
2. **Who should guard whom.** A matchup model predicts points per possession for any defender vs. any scorer, and the **Hungarian algorithm** finds the defensive assignment (and lineup) that gives up the fewest points against a specific opponent.

**Live dashboard:** https://anishatl.github.io/nba-lineup-matchup-lab/ *(in progress)*

---

## How it works

```
NBA.com play-by-play ─► stints (same 10 players on floor) ─► RAPM (ridge regression) ─┐
                                                                                       ├─► lineup score ─► static dashboard
NBA.com matchup tracking ─► who-guarded-whom ─► matchup model ─► Hungarian assignment ─┘
```

All raw data lives in a local **PostgreSQL** database. Models run locally, then export small JSON files into `docs/data/`, which GitHub Pages serves as a static site.

### Lineup score

Everything is in **points per 100 possessions**, so offense and defense combine directly:

```
expected scored  = intercept + Σ O-RAPM (your 5) + Σ D-RAPM (their 5)
expected allowed = 100 × Σ usage_share(scorer) × predicted PPP(defender vs scorer)   ← Hungarian-optimized
net              = expected scored − slider × expected allowed                       (slider defaults to 1.0)
```

The two components are calibrated against real stint outcomes so they're on the same scale.

### Sign convention (important)

| Where | O-RAPM | D-RAPM |
|---|---|---|
| Code & database (raw regression coefficients) | positive = good | **negative = good** (fewer points allowed) |
| Dashboard display only | positive = good | **flipped: positive = good** |

All math uses the raw coefficients. The flip happens only when rendering.

---

## Setup

**Requirements:** Python 3.10+, PostgreSQL 15+.

```bash
git clone https://github.com/Anishatl/nba-lineup-matchup-lab.git
cd nba-lineup-matchup-lab

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

# Create the database (psql comes with PostgreSQL)
createdb -U postgres nba_lab

# Add your Postgres login
cp .env.example .env    # Windows: copy .env.example .env
# then edit .env and put in your password
```

## Run the pipeline

```bash
# Quick test first: 5 games from one season (~1 minute)
python scripts/run_pipeline.py --seasons 2025-26 --limit 5

# Full pull: all default seasons (2023-24 → current). Takes a few hours.
python scripts/run_pipeline.py
```

- **Resumable:** stop it anytime (Ctrl+C, close the laptop) and re-run the same command. Finished games are skipped, and failed games are retried.
- **Cached:** raw responses are saved in `data/raw/`, so nothing is downloaded twice.
- **Logs:** `logs/pipeline.log`.
- Must run from a home/residential connection. NBA.com blocks most cloud servers.

| Step | Source | Calls |
|---|---|---|
| `meta` | teams, player bios, game logs | ~10 total |
| `stints` | NBA liveData play-by-play via `pbpstats` | 1 per game |
| `matchups` | `BoxScoreMatchupsV3` | 1 per game |

## Starter SQL

```bash
psql -U postgres -d nba_lab -v team=ATL -v season=2025-26 -v min_poss=150 -f db/queries/best_lineups.sql
psql -U postgres -d nba_lab -v team=ATL -v season=2025-26 -f db/queries/player_on_off.sql
psql -U postgres -d nba_lab -v season=2025-26 -v min_poss=1500 -v team=ALL -f db/queries/top_defenders.sql
```

Swap `ATL` for any team abbreviation.

## Database

| Table | One row per |
|---|---|
| `teams` | team |
| `players` | player (name, position, height, weight) |
| `player_team_season` | player × team × season (handles trades) |
| `games` | game (date, teams, score) |
| `stints` | stretch of possessions with the same 10 players on the floor |
| `matchups` | game × offensive player × defender (tracked possessions, points, shots) |
| `pipeline_log` | game × step progress, for resuming |

## Project structure

```
config.py              settings (seasons, request pacing, paths)
db/schema.sql          tables
db/queries/            starter SQL
src/                   data pull + (later) models and export
scripts/run_pipeline.py
docs/                  static dashboard (GitHub Pages)
data/raw/              cached raw responses (git-ignored)
```

## Roadmap

- [x] **Step 1: Data pipeline.** Postgres schema, play-by-play → stints, matchup tracking, starter SQL
- [ ] **Step 2: RAPM.** Ridge regression on stints, cross-validated penalty, out-of-sample check vs. raw +/-
- [ ] **Step 3: Matchup model.** Shrunk defender-vs-scorer PPP, model for unseen pairs, Hungarian assignment, calibration
- [ ] **Step 4: Dashboard.** Team picker, lineup explorer, matchup planner with defensive-emphasis slider, player cards

## Known limitations

- Matchup data is camera-tracked and gets noisy on switches and help defense.
- Possessions where a substitution happens mid-possession (e.g. between free throws) are credited to the lineup that started the possession.
- RAPM needs a lot of data. Players with few minutes are pulled strongly toward average.
