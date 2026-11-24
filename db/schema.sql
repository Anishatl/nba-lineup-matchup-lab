-- NBA Lineup & Matchup Lab: database schema
-- Safe to run repeatedly (everything is IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS teams (
    team_id       INTEGER PRIMARY KEY,
    abbreviation  TEXT NOT NULL,
    city          TEXT,
    name          TEXT
);

CREATE TABLE IF NOT EXISTS players (
    player_id      INTEGER PRIMARY KEY,
    full_name      TEXT NOT NULL,
    position       TEXT,
    height_inches  INTEGER,
    weight_lbs     INTEGER
);

-- Who played for which team each season (handles mid-season trades:
-- a traded player gets one row per team).
CREATE TABLE IF NOT EXISTS player_team_season (
    player_id  INTEGER REFERENCES players(player_id),
    team_id    INTEGER REFERENCES teams(team_id),
    season     TEXT NOT NULL,
    games      INTEGER,
    minutes    NUMERIC,
    PRIMARY KEY (player_id, team_id, season)
);

CREATE TABLE IF NOT EXISTS games (
    game_id       TEXT PRIMARY KEY,
    season        TEXT NOT NULL,
    game_date     DATE,
    home_team_id  INTEGER REFERENCES teams(team_id),
    away_team_id  INTEGER REFERENCES teams(team_id),
    home_pts      INTEGER,
    away_pts      INTEGER
);
CREATE INDEX IF NOT EXISTS games_season_idx ON games (season);

-- A stint = consecutive possessions in one period with the same 10 players
-- on the floor. Player arrays are sorted so identical lineups group together.
-- This is the input to the RAPM model.
CREATE TABLE IF NOT EXISTS stints (
    game_id        TEXT REFERENCES games(game_id),
    stint_num      INTEGER,
    period         INTEGER,
    home_players   INTEGER[] NOT NULL,
    away_players   INTEGER[] NOT NULL,
    seconds        NUMERIC,
    home_poss      INTEGER,   -- possessions where home team had the ball
    away_poss      INTEGER,
    home_pts       INTEGER,   -- points home team scored during the stint
    away_pts       INTEGER,
    PRIMARY KEY (game_id, stint_num)
);
CREATE INDEX IF NOT EXISTS stints_home_players_idx ON stints USING GIN (home_players);
CREATE INDEX IF NOT EXISTS stints_away_players_idx ON stints USING GIN (away_players);

-- Who guarded whom (NBA.com tracking, BoxScoreMatchupsV3).
-- Input to the matchup model.
CREATE TABLE IF NOT EXISTS matchups (
    game_id           TEXT REFERENCES games(game_id),
    off_team_id       INTEGER,
    off_player_id     INTEGER,
    def_player_id     INTEGER,
    matchup_seconds   NUMERIC,
    partial_poss      NUMERIC,
    player_pts        INTEGER,   -- points the offensive player scored vs this defender
    team_pts          INTEGER,   -- points the offense scored while matched up
    fgm               INTEGER,
    fga               INTEGER,
    fg3m              INTEGER,
    fg3a              INTEGER,
    ftm               INTEGER,
    fta               INTEGER,
    ast               INTEGER,
    tov               INTEGER,
    blk               INTEGER,
    help_blk          INTEGER,
    switches_on       INTEGER,
    shooting_fouls    INTEGER,
    PRIMARY KEY (game_id, off_player_id, def_player_id)
);
CREATE INDEX IF NOT EXISTS matchups_def_idx ON matchups (def_player_id);
CREATE INDEX IF NOT EXISTS matchups_off_idx ON matchups (off_player_id);

-- Tracks which per-game steps finished, so the pipeline can resume after a crash.
CREATE TABLE IF NOT EXISTS pipeline_log (
    game_id    TEXT,
    step       TEXT,          -- 'stints' or 'matchups'
    status     TEXT,          -- 'done' or 'failed'
    detail     TEXT,
    updated_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (game_id, step)
);

-- ---------------------------------------------------------------------------
-- Step 2: RAPM
-- ---------------------------------------------------------------------------

-- One row per model fit (re-run weekly and keep the history).
CREATE TABLE IF NOT EXISTS model_runs (
    run_id          SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT now(),
    model           TEXT NOT NULL,          -- 'rapm'
    seasons         TEXT[] NOT NULL,
    season_weights  NUMERIC[],
    alpha           NUMERIC,                -- ridge penalty (lambda)
    intercept       NUMERIC,                -- league-average pts per 100
    home_adv        NUMERIC,                -- home-court pts per 100
    n_rows          INTEGER,
    n_players       INTEGER,
    cv_rmse         NUMERIC,
    notes           TEXT
);

-- Player ratings for a run. Raw regression coefficients:
--   o_rapm > 0 is good, d_rapm < 0 is good (fewer points allowed).
--   total_rapm = o_rapm - d_rapm (net impact, > 0 is good).
CREATE TABLE IF NOT EXISTS player_ratings (
    run_id      INTEGER REFERENCES model_runs(run_id) ON DELETE CASCADE,
    player_id   INTEGER REFERENCES players(player_id),
    o_rapm      NUMERIC,
    d_rapm      NUMERIC,
    total_rapm  NUMERIC,
    off_poss    INTEGER,
    def_poss    INTEGER,
    PRIMARY KEY (run_id, player_id)
);

-- ---------------------------------------------------------------------------
-- Step 3: matchup model
-- ---------------------------------------------------------------------------

-- Per-player skills from the matchup model (model_runs.model = 'matchup').
-- Units: points per 100 matchup possessions vs. the model intercept.
--   scorer_skill   > 0 = scores more than average against everyone
--   defender_skill < 0 = holds scorers below average (good defender)
-- Height/weight/three_rate are the inputs used for the size/style terms.
CREATE TABLE IF NOT EXISTS matchup_ratings (
    run_id          INTEGER REFERENCES model_runs(run_id) ON DELETE CASCADE,
    player_id       INTEGER REFERENCES players(player_id),
    scorer_skill    NUMERIC,
    defender_skill  NUMERIC,
    off_poss        NUMERIC,
    def_poss        NUMERIC,
    height_inches   NUMERIC,
    weight_lbs      NUMERIC,
    three_rate      NUMERIC,
    PRIMARY KEY (run_id, player_id)
);

-- Named scalar parameters for a run (feature weights, centering constants,
-- calibration a/b/c, ...). Lets the dashboard rebuild any prediction.
CREATE TABLE IF NOT EXISTS model_params (
    run_id  INTEGER REFERENCES model_runs(run_id) ON DELETE CASCADE,
    name    TEXT,
    value   NUMERIC,
    PRIMARY KEY (run_id, name)
);
