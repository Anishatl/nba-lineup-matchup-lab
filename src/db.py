"""Postgres connection + bulk upsert helpers."""
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import execute_values

import config


@contextmanager
def get_conn():
    conn = psycopg2.connect(config.DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema() -> None:
    """Create all tables (safe to run repeatedly)."""
    sql = config.SCHEMA_FILE.read_text(encoding="utf-8")
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)


def upsert(conn, table: str, columns: list[str], rows: list[tuple],
           conflict: list[str], update: list[str] | None = None) -> int:
    """Insert rows; on primary-key conflict either skip (update=None) or
    overwrite the listed columns. Returns number of rows sent."""
    if not rows:
        return 0
    cols = ", ".join(columns)
    conflict_cols = ", ".join(conflict)
    if update:
        action = "DO UPDATE SET " + ", ".join(f"{c} = EXCLUDED.{c}" for c in update)
    else:
        action = "DO NOTHING"
    sql = f"INSERT INTO {table} ({cols}) VALUES %s ON CONFLICT ({conflict_cols}) {action}"
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=1000)
    return len(rows)


def done_games(conn, step: str) -> set[str]:
    """Game IDs that already finished a per-game step."""
    with conn.cursor() as cur:
        cur.execute("SELECT game_id FROM pipeline_log WHERE step = %s AND status = 'done'", (step,))
        return {r[0] for r in cur.fetchall()}


def mark(conn, game_id: str, step: str, status: str, detail: str = "") -> None:
    upsert(conn, "pipeline_log", ["game_id", "step", "status", "detail"],
           [(game_id, step, status, detail[:500])],
           conflict=["game_id", "step"], update=["status", "detail"])
