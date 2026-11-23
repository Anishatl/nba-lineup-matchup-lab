"""Shared helpers: logging, polite pacing, retries, JSON caching."""
import json
import logging
import time
from pathlib import Path

import config

config.LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_DIR / "pipeline.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


log = get_logger("utils")
_last_request = 0.0


def throttle() -> None:
    """Wait so consecutive NBA.com requests are at least REQUEST_DELAY apart."""
    global _last_request
    wait = config.REQUEST_DELAY - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def with_retries(fn, *args, retries: int = config.MAX_RETRIES, **kwargs):
    """Call fn, retrying with exponential backoff on any exception
    (NBA.com often times out or drops connections)."""
    for attempt in range(1, retries + 1):
        throttle()
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - we really do want to retry anything
            if attempt == retries:
                raise
            backoff = 2 ** attempt
            log.warning("%s failed (%s), retry %d/%d in %ds",
                        getattr(fn, "__name__", fn), exc, attempt, retries, backoff)
            time.sleep(backoff)


def read_json(path: Path):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def clock_to_seconds(value) -> float:
    """'5:32' or '05:32.00' or 'PT05M32.00S' -> 332.0. Numbers pass through."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s.startswith("PT"):
        s = s[2:].rstrip("S").replace("M", ":")
    if ":" in s:
        minutes, seconds = s.split(":", 1)
        return int(minutes or 0) * 60 + float(seconds or 0)
    return float(s) if s else 0.0


def height_to_inches(h):
    """'6-8' -> 80. Returns None if unknown."""
    try:
        feet, inches = str(h).split("-")
        return int(feet) * 12 + int(inches)
    except (ValueError, AttributeError):
        return None


def to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None
