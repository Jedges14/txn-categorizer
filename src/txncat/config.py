"""Settings from environment variables (a local `.env` file is loaded if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_URL = "postgresql+psycopg://txncat:txncat@localhost:5433/txncat"


@dataclass(frozen=True)
class Settings:
    database_url: str
    connect_timeout_s: int


def get_settings() -> Settings:
    # Real environment variables win over .env (override=False).
    load_dotenv(REPO_ROOT / ".env", override=False)
    return Settings(
        database_url=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
        connect_timeout_s=int(os.environ.get("DB_CONNECT_TIMEOUT_S", "5")),
    )
