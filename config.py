"""Project-wide settings. Change seasons, request pacing, and paths here."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Seasons to pull by default (NBA.com format). The current season is included
# so the pipeline can be re-run weekly to pick up new games.
SEASONS = ["2023-24", "2024-25", "2025-26", "2026-27"]
SEASON_TYPE = "Regular Season"

# Seconds to wait between NBA.com stats requests. Going faster gets you blocked.
REQUEST_DELAY = 0.7
REQUEST_TIMEOUT = 60
MAX_RETRIES = 5

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres@localhost:5432/nba_lab")

# Raw responses are cached here so nothing is downloaded twice.
RAW_DIR = ROOT / "data" / "raw"
PBP_DIR = RAW_DIR / "pbpstats"        # pbpstats play-by-play + boxscore JSON
MATCHUP_DIR = RAW_DIR / "matchups"    # BoxScoreMatchupsV3 JSON
LOG_DIR = ROOT / "logs"

SCHEMA_FILE = ROOT / "db" / "schema.sql"

# Static dashboard (GitHub Pages serves the docs/ folder)
SITE_DATA_DIR = ROOT / "docs" / "data"
