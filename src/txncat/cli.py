"""`txncat` command-line entry point.

Commands so far: `db check|init` (M0), `data build|load|report` (M1). Later milestones add `train baseline`
(M2), `train finetune` (M3), `llm run` (M4), `bench latency` and `report` (M5).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

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

    train = sub.add_parser("train", help="model training")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    train_sub.add_parser(
        "baseline", help="E0 majority/rules, E1 logistic TF-IDF, E2 XGBoost TF-IDF+SQL; tuned on val"
    ).set_defaults(func=_train_baseline)
    ft = train_sub.add_parser("finetune", help="E3 fine-tuned DistilBERT (needs: uv sync --extra encoder)")
    ft.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ft.add_argument("--epochs", type=int, default=3)
    ft.add_argument("--input", type=Path, default=None, help="CSV from `txncat data export-encoder` (e.g. on Colab)")
    ft.set_defaults(func=_train_finetune)

    data_sub.add_parser(
        "export-encoder", help="write data/generated/encoder_input.csv for training E3 elsewhere (Colab)"
    ).set_defaults(func=_data_export_encoder)

    llm = sub.add_parser("llm", help="E4/E5 with a free local open-weight LLM (no API key, no cost)")
    llm_sub = llm.add_subparsers(dest="llm_command", required=True)
    lr = llm_sub.add_parser("run", help="classify with the LLM; needs: uv sync --extra encoder")
    lr.add_argument("--mode", choices=["zero", "few"], required=True, help="zero = E4, few = E5 (3 per category)")
    lr.add_argument("--splits", nargs="+", default=["test", "external_test"])
    lr.add_argument("--limit", type=int, default=None, help="random sample per split (pilot runs)")
    lr.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    lr.add_argument("--threads", type=int, default=None)
    lr.set_defaults(func=_llm_run)

    bench = sub.add_parser("bench", help="benchmarks")
    bench_sub = bench.add_subparsers(dest="bench_command", required=True)
    bench_sub.add_parser(
        "latency", help="per-prediction latency of E1-E3 (batch size 1) -> reports/metrics/latency.json"
    ).set_defaults(func=_bench_latency)

    rep = sub.add_parser("report", help="M5 tables, significance tests and figures from saved predictions")
    rep.add_argument("--n-boot", type=int, default=1000)
    rep.add_argument("--no-figures", action="store_true")
    rep.set_defaults(func=_report)

    sub.add_parser(
        "errors", help="M6 error analysis of the best external-test model (needs the database)"
    ).set_defaults(func=_errors)
    return parser


def _errors(_: argparse.Namespace) -> int:
    from txncat.eval.errors_cli import run

    try:
        return run()
    except OperationalError as exc:
        print(f"Cannot reach PostgreSQL: {exc.orig}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"cannot build the error analysis: {exc}", file=sys.stderr)
        return 1


def _report(args: argparse.Namespace) -> int:
    from txncat.eval.report import run

    try:
        return run(n_boot=args.n_boot, make_figures=not args.no_figures)
    except (FileNotFoundError, ValueError) as exc:
        print(f"cannot build the report: {exc}", file=sys.stderr)
        return 1


def _bench_latency(_: argparse.Namespace) -> int:
    import json
    from datetime import UTC, datetime

    from txncat.eval.latency import measure
    from txncat.features.sql_features import load_features
    from txncat.models.train_cli import _git_commit

    try:
        engine = make_engine()
        apply_schema(engine)
        df = load_features(engine)
    except OperationalError as exc:
        print(f"Cannot reach PostgreSQL: {exc.orig}", file=sys.stderr)
        return 1
    import platform

    result = {
        "command": "txncat bench latency",
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "machine": {"processor": platform.processor(), "platform": platform.platform()},
        **measure(df),
    }
    path = REPO_ROOT / "reports" / "metrics" / "latency.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"saved {path.relative_to(REPO_ROOT)}")
    return 0


def _llm_run(args: argparse.Namespace) -> int:
    try:
        from txncat.models.llm_cli import run
    except ImportError as exc:
        print(f"{exc}. Install the encoder extra: uv sync --extra encoder", file=sys.stderr)
        return 1
    try:
        return run(args.mode, args.splits, args.limit, args.model, args.threads)
    except OperationalError as exc:
        print(f"Cannot reach PostgreSQL: {exc.orig}", file=sys.stderr)
        return 1


def _train_finetune(args: argparse.Namespace) -> int:
    try:
        from txncat.models.finetune_cli import run
    except ImportError as exc:
        print(f"{exc}. Install the encoder extra: uv sync --extra encoder", file=sys.stderr)
        return 1
    return run(args.seeds, args.input, args.epochs)


def _data_export_encoder(_: argparse.Namespace) -> int:
    from txncat.models.finetune_cli import export_encoder_input

    path = export_encoder_input()
    print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


def _train_baseline(_: argparse.Namespace) -> int:
    from txncat.models.train_cli import run

    try:
        return run()
    except OperationalError as exc:
        print(f"Cannot reach PostgreSQL: {exc.orig}", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
