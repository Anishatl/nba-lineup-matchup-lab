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

## Fit the RAPM model

```bash
python scripts/run_rapm.py --validate
```

- Builds the design matrix (2 rows per stint, O and D column per player, home-court column).
- Picks the ridge penalty with 5-fold cross-validation **split by game** (no game's stints in both train and test).
- Recency-weights seasons (current ×1, last ×0.5, ...). Change with `--decay`.
- Saves the run to `model_runs` and ratings to `player_ratings`, prints top/bottom players.
- `--validate` trains on 2023-24 + 2024-25, predicts 2025-26, compares RAPM against league average and raw +/-, and writes `reports/rapm_validation.md`.

```bash
psql -U postgres -d nba_lab -v team=ATL -v min_poss=1000 -f db/queries/player_ratings.sql
```

## Fit the matchup model

```bash
python scripts/run_matchups.py                # Poisson (default): fit + validate + calibrate + save
python scripts/run_matchups.py --kind linear  # additive version, for comparison
```
Run `run_rapm.py` first.

- **Matchup model (ridge Poisson regression).** For every game × (scorer, defender) row:
  `log(pts per matchup possession) = intercept + scorer skill + defender skill + size/style terms`,
  so `pts = base × scorer multiplier × defender multiplier × size multiplier`.
  A defender takes away a *share* of whoever he guards (e.g. ×0.88 = −12%), so he matters more against
  high-volume scorers. That is what makes who-guards-whom matter: in an additive model every assignment
  of the same 5 defenders gives the same total, so there is nothing to optimize.
- **Size/style terms** (true pair interactions, centered on the average matchup): squared height mismatch,
  and scorer 3PA share × defender height. Additive height/weight differences are left out because the
  player skills absorb them and the skills become hard to read.
- **Team defensive context (fit only).** Each row also gets a term for the *defending team-season*,
  so help defense, rim protection and scheme are credited to the team instead of to whichever player
  was nominally guarding. Predictions and the optimizer use player terms only (a team term scales every
  matchup for that team equally, so it can't change which lineup or assignment is best). An offensive
  team-season term was tested on synthetic data and dropped: it took credit away from star scorers.
  `--no-context` turns it off.
- **Rankings** use each player's *total* effect (size terms included) against a typical mix of opponents,
  as a percentage.
- Rows are aggregated to (scorer, defender, season) for fitting (identical result, much faster);
  cross-validation still splits raw rows by game. Poisson is scored by deviance.
- **Validation:** trains on 2023-24 + 2024-25, predicts 2025-26 matchups; compares league average,
  the linear model, and Poisson with scorer only / + defender / + size-style / + team context.
- **Calibration:** on the held-out season, fits
  `pts/100 = base + a·ΣO-RAPM + b·(optimal matchup defense) + c·ΣD-RAPM + h·home`
  so both models are on one scale, and compares against RAPM alone.
- Points come in 1s, 2s and 3s, so they vary more than a Poisson model assumes (overdispersion).
  That doesn't bias the estimates, only their standard errors (not reported; a game-level bootstrap
  would be the way to add them).
- Writes `reports/matchup_validation.md`.

## Plan a matchup

```bash
python scripts/plan_matchup.py --team ATL --opp NYK
python scripts/plan_matchup.py --team ATL --opp NYK --slider 1.5 --size-strength 0.5
python scripts/plan_matchup.py --team ATL --opp BOS --opp-players Tatum Brown White Holiday Porzingis
```

Ranks every 5-man combination of your roster against the opponent's lineup (default: their most-used
five this season) and shows the **two best defensive assignments** for the top lineup, with the gap
between them. Players can be passed by name or ID on either side.

- `--slider` (defensive emphasis): `net = scored − slider × allowed` (pts/100 vs. an average matchup).
- `--size-strength`: scales the size/style matchup effects (1 = as fitted, 0 = off), to see how much a
  recommendation depends on them. Size mismatches in the data often happen on switches, so the fitted
  effect may partly reflect switching situations rather than size alone.

**Sandbox:** score your own five and your own assignments, and see what they cost vs. the optimum:

```bash
python scripts/plan_matchup.py --team ATL --opp NYK \
    --my-players Daniels McCollum Johnson Alexander-Walker Okongwu \
    --assign Daniels=Brunson McCollum=Bridges Johnson=Anunoby Alexander-Walker=Hart Okongwu=Towns
```

Points allowed use your assignment; points scored assume the opponent defends your five optimally.
The calibration weight on matchup defense is constrained to be ≥ 0 (a negative weight would mean worse
matchups lead to fewer points allowed).

## Starter SQL

```bash
psql -U postgres -d nba_lab -v team=ATL -v season=2025-26 -v min_poss=150 -f db/queries/best_lineups.sql
psql -U postgres -d nba_lab -v team=ATL -v season=2025-26 -f db/queries/player_on_off.sql
psql -U postgres -d nba_lab -v season=2025-26 -v min_poss=1500 -v team=ALL -f db/queries/top_defenders.sql
psql -U postgres -d nba_lab -v team=ALL -v min_poss=1500 -f db/queries/matchup_ratings.sql
```

Swap `ATL` for any team abbreviation.

**Windows:** if psql errors with `has no equivalent in encoding "WIN1252"` (player names like Porziņģis), switch the console to UTF-8 first:
```powershell
chcp 65001
$env:PGCLIENTENCODING = "UTF8"
```

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

- [x] **Step 1: Data pipeline.** 3,934 games validated against box scores. Postgres schema, play-by-play → stints, matchup tracking, starter SQL
- [x] **Step 2: RAPM.** Ridge regression on stints, cross-validated penalty, out-of-sample check vs. raw +/-
- [x] **Step 3: Matchup model.** Shrunk defender-vs-scorer PPP, model for unseen pairs, Hungarian assignment, calibration
- [ ] **Step 4: Dashboard.** Team picker, lineup explorer, matchup planner with defensive-emphasis slider, player cards

## Data coverage & validation

Stint points are checked against official box scores for every game (`scripts/check_stints.py`).

| Season | Games | Stints loaded | Exact score match |
|---|---|---|---|
| 2023-24 | 1,230 | 1,230 | 1,230 |
| 2024-25 | 1,225 | 1,224 | 1,223 |
| 2025-26 | 1,225 | 1,225 | 1,224 |
| 2026-27 (through Nov 23) | 257 | 255 | 253 |
| **Total** | **3,937** | **3,934 (99.9%)** | **3,930 (99.8%)** |

Matchup tracking: 3,922 games (the most recent games are added once NBA.com posts them).

Problems found and fixed along the way:
- **Points short by ~5%.** Lineups were read from the first event of each possession, which is sometimes incomplete. Fixed by scanning every event in the possession. Result: 0 dropped possessions.
- **Overtime starters unknown.** A player on the floor with no recorded events is invisible to the play-by-play. Fixed by looking up who played in the first seconds of the period in NBA.com's time-ranged box score and saving it as a pbpstats override.
- **2026-27 feed format changes.** NBA's live feed began omitting `personId` on non-player events and sometimes references rebounds to nonexistent shot IDs. Patched in `src/pbpstats_patches.py`.
- **3 games excluded** for malformed source data (a foul and a violation missing fields, and a team rebound with no preceding miss).

## Known limitations

- Matchup data is camera-tracked and gets noisy on switches and help defense.
- Possessions where a substitution happens mid-possession (e.g. between free throws) are credited to the lineup that started the possession.
- RAPM needs a lot of data. Players with few minutes are pulled strongly toward average.
