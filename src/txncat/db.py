"""PostgreSQL access: engine creation, connectivity check, schema setup."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, text

from txncat.config import REPO_ROOT, get_settings

SQL_DIR = REPO_ROOT / "sql"
SCHEMA_FILES = ("001_schema.sql", "002_external_test.sql")


def make_engine(url: str | None = None) -> Engine:
    settings = get_settings()
    # Without a timeout, an unreachable server can hang for minutes (notably on Windows).
    return create_engine(
        url or settings.database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": settings.connect_timeout_s},
    )


def server_version(engine: Engine) -> str:
    with engine.connect() as conn:
        return str(conn.execute(text("SHOW server_version")).scalar_one())


def apply_schema(engine: Engine, files: tuple[str, ...] = SCHEMA_FILES) -> list[Path]:
    """Run the numbered SQL files in order. They are idempotent (IF NOT EXISTS)."""
    applied: list[Path] = []
    with engine.begin() as conn:
        for name in files:
            path = SQL_DIR / name
            conn.exec_driver_sql(path.read_text(encoding="utf-8"))
            applied.append(path)
    return applied
