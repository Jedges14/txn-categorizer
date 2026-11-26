"""`txncat` command-line entry point.

Commands so far: `db check|init` (M0), `data build|load|report` (M1). Later milestones add `train baseline`
(M2), `train finetune` (M3), `llm run` (M4) and `report` (M5).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from sqlalchemy.exc import OperationalError

from txncat import __version__
from txncat.config import REPO_ROOT, get_settings
from txncat.db import apply_schema, make_engine, server_version

MERCHANTS_CSV = REPO_ROOT / "data" / "merchants.csv"
SYNTHETIC_CSV = REPO_ROOT / "data" / "generated" / "synthetic.csv"
REAL_TEST_CSV = REPO_ROOT / "data" / "private" / "real_test.csv"
FIGURE = REPO_ROOT / "reports" / "figures" / "class_distribution.png"
STATS_MD = REPO_ROOT / "reports" / "data_summary.md"


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


def _data_build(args: argparse.Namespace) -> int:
    from txncat.data.generate_synthetic import generate, write_csv
    from txncat.data.merchants import build_merchants, write_merchants_csv

    merchants = build_merchants(seed=args.seed)
    write_merchants_csv(merchants, MERCHANTS_CSV)
    rows = generate(merchants, n_rows=args.rows, seed=args.seed)
    write_csv(rows, SYNTHETIC_CSV)
    print(f"wrote {len(merchants)} merchants → {MERCHANTS_CSV.relative_to(REPO_ROOT)}")
    print(f"wrote {len(rows)} transactions (seed {args.seed}) → {SYNTHETIC_CSV.relative_to(REPO_ROOT)}")
    return 0


def _data_load(_: argparse.Namespace) -> int:
    from txncat.data.load_to_postgres import (
        RealTestError,
        read_real_test_csv,
        read_synthetic_csv,
        replace_source,
    )

    if not SYNTHETIC_CSV.exists():
        print("No synthetic data yet. Run: txncat data build", file=sys.stderr)
        return 1
    try:
        engine = make_engine()
        apply_schema(engine)
        s = replace_source(engine, "synthetic", read_synthetic_csv(SYNTHETIC_CSV))
        print(f"loaded {s.rows} synthetic rows: {s.by_split}")
        if REAL_TEST_CSV.exists():
            try:
                real_rows = read_real_test_csv(REAL_TEST_CSV)
            except RealTestError as exc:
                print(
                    f"data/private/real_test.csv has problems (nothing loaded from it):\n{exc}",
                    file=sys.stderr,
                )
                return 1
            r = replace_source(engine, "real", real_rows)
            print(f"loaded {r.rows} real_test rows from data/private/real_test.csv")
        else:
            print("no data/private/real_test.csv (optional; template: data/sample/real_test_example.csv)")
        from txncat.data.external import LOCAL_PATH, SOURCE, build_external_test

        if LOCAL_PATH.exists():
            e = replace_source(engine, SOURCE, build_external_test())
            print(f"loaded {e.rows} external_test rows ({SOURCE})")
        else:
            print("no external test set yet (run: txncat data external)")
    except OperationalError as exc:
        print(f"Cannot reach PostgreSQL: {exc.orig}", file=sys.stderr)
        return 1
    return 0


def _data_external(_: argparse.Namespace) -> int:
    from txncat.data.external import LOCAL_PATH, REPO_ID, REVISION, ExternalDataError, download

    try:
        path = download()
    except (ExternalDataError, OSError) as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {REPO_ID}@{REVISION[:8]} (checksum verified) → {path.relative_to(REPO_ROOT)}")
    print("next: txncat data load")
    return 0 if path == LOCAL_PATH else 1


def _data_report(_: argparse.Namespace) -> int:
    from txncat.data.report import class_distribution_chart, counts_from_csv, markdown_table

    if not SYNTHETIC_CSV.exists():
        print("No synthetic data yet. Run: txncat data build", file=sys.stderr)
        return 1
    rows, merchants = counts_from_csv(SYNTHETIC_CSV)
    table = markdown_table(rows, merchants)
    STATS_MD.parent.mkdir(parents=True, exist_ok=True)
    STATS_MD.write_text(
        "# Synthetic data summary\n\nGenerated by `txncat data build` and `txncat data report`.\n\n" + table + "\n",
        encoding="utf-8",
    )
    chart = class_distribution_chart(rows, FIGURE)
    from txncat.data.external import LOCAL_PATH, build_external_test, overlap_with_merchants

    if LOCAL_PATH.exists():
        import csv
        from collections import Counter

        with SYNTHETIC_CSV.open(newline="", encoding="utf-8") as f:
            train_merchants = {r["merchant"] for r in csv.DictReader(f) if r["split"] == "train"}
        ext = build_external_test()
        seen = overlap_with_merchants(ext, train_merchants)
        n_seen = sum(s is not None for s in seen)
        per_cat = Counter(r["category"] for r in ext)
        seen_cat = Counter(r["category"] for r, s in zip(ext, seen, strict=True) if s is not None)
        ext_lines = [
            "",
            "## External test set (DoDataThings v2, mapped)",
            "",
            f"{len(ext)} rows; {n_seen} ({n_seen / len(ext):.1%}) mention a merchant name from the TRAIN split.",
            "",
            "| Category | rows | mention a train merchant |",
            "|---|---|---|",
            *(f"| {c} | {per_cat[c]} | {seen_cat[c]} |" for c in sorted(per_cat)),
        ]
        table += "\n" + "\n".join(ext_lines)
        STATS_MD.write_text(
            "# Data summary\n\nGenerated by `txncat data build`, `txncat data external` and `txncat data report`.\n\n"
            + table
            + "\n",
            encoding="utf-8",
        )
    print(table)
    print(f"\nsaved {chart.relative_to(REPO_ROOT)} and {STATS_MD.relative_to(REPO_ROOT)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="txncat", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"txncat {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    db = sub.add_parser("db", help="database commands")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    db_sub.add_parser("check", help="check the PostgreSQL connection").set_defaults(func=_db_check)
    db_sub.add_parser("init", help="create tables (idempotent)").set_defaults(func=_db_init)

    data = sub.add_parser("data", help="dataset commands")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    build = data_sub.add_parser("build", help="write data/merchants.csv and synthetic transactions")
    build.add_argument("--rows", type=int, default=20_000)
    build.add_argument("--seed", type=int, default=42)
    build.set_defaults(func=_data_build)
    data_sub.add_parser("load", help="load synthetic (+ private real test) rows into Postgres").set_defaults(
        func=_data_load
    )
    data_sub.add_parser("external", help="download the pinned external test dataset (MIT, Hugging Face)").set_defaults(
        func=_data_external
    )
    data_sub.add_parser("report", help="class distribution table and chart").set_defaults(func=_data_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
