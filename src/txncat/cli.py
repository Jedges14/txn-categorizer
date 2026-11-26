"""`txncat` command-line entry point.

M0 ships the database commands. Later milestones add: `data build|load` (M1), `train baseline` (M2),
`train finetune` (M3), `llm run` (M4) and `report` (M5).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from sqlalchemy.exc import OperationalError

from txncat import __version__
from txncat.config import get_settings
from txncat.db import apply_schema, make_engine, server_version


def _redact(url: str) -> str:
    """Hide the password in a database URL before printing it."""
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host = rest.rsplit("@", 1)
    user = creds.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"


def _db_check(_: argparse.Namespace) -> int:
    url = get_settings().database_url
    try:
        version = server_version(make_engine(url))
    except OperationalError as exc:
        print(f"Cannot reach PostgreSQL at {_redact(url)}: {exc.orig}", file=sys.stderr)
        print("Start it with: docker compose up -d", file=sys.stderr)
        return 1
    print(f"OK: PostgreSQL {version} at {_redact(url)}")
    return 0


def _db_init(_: argparse.Namespace) -> int:
    try:
        applied = apply_schema(make_engine())
    except OperationalError as exc:
        print(f"Cannot reach PostgreSQL: {exc.orig}", file=sys.stderr)
        return 1
    for path in applied:
        print(f"applied {path.name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="txncat", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"txncat {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    db = sub.add_parser("db", help="database commands")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    db_sub.add_parser("check", help="check the PostgreSQL connection").set_defaults(func=_db_check)
    db_sub.add_parser("init", help="create tables (idempotent)").set_defaults(func=_db_init)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
