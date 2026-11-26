"""`txncat train baseline`: run E0–E2, score every split, save predictions + metrics (for M5's `txncat report`)."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from txncat.config import REPO_ROOT
from txncat.data.external import overlap_with_merchants
from txncat.db import apply_schema, make_engine
from txncat.eval.metrics import score
from txncat.features.sql_features import load_features
from txncat.models.baseline import ModelResult, run_baselines

PRED_DIR = REPO_ROOT / "predictions"
METRICS_PATH = REPO_ROOT / "reports" / "metrics" / "baseline.json"
SYNTHETIC_CSV = REPO_ROOT / "data" / "generated" / "synthetic.csv"


def _git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{sha}+uncommitted" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def external_seen_ids(df: pd.DataFrame) -> set[int]:
    """IDs of external_test rows that mention a TRAIN-split merchant name (reported separately)."""
    if not SYNTHETIC_CSV.exists():
        return set()
    with SYNTHETIC_CSV.open(newline="", encoding="utf-8") as f:
        train_merchants = {r["merchant"] for r in csv.DictReader(f) if r["split"] == "train"}
    ext = df[df["split"] == "external_test"]
    seen = overlap_with_merchants(ext[["description"]].to_dict("records"), train_merchants)
    return {int(i) for i, s in zip(ext["id"], seen, strict=True) if s is not None}


def evaluate(result: ModelResult, seen_ext: set[int]) -> dict:
    p = result.predictions
    out: dict = {}
    for split in ("val", "test", "external_test"):
        part = p[p["split"] == split]
        if len(part):
            out[split] = score(part["category"], part["pred"]).as_dict()
    ext = p[p["split"] == "external_test"]
    unseen = ext[~ext["id"].isin(seen_ext)]
    if len(unseen) and seen_ext:
        out["external_test_unseen_merchants"] = score(unseen["category"], unseen["pred"]).as_dict()
    return out


def run() -> int:
    engine = make_engine()
    apply_schema(engine)
    df = load_features(engine)
    started = time.perf_counter()
    results = run_baselines(df)
    seen_ext = external_seen_ids(df)

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "command": "txncat train baseline",
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "rows": {s: int((df["split"] == s).sum()) for s in ("train", "val", "test", "external_test")},
        "external_seen_merchant_rows": len(seen_ext),
        "models": {},
    }
    for r in results:
        r.predictions.to_csv(PRED_DIR / f"{r.name}.csv", index=False)
        report["models"][r.name] = {
            "params": r.params,
            "val_macro_f1_selected": r.val_macro_f1,
            "fit_seconds": round(r.fit_seconds, 1),
            "tuning": r.tuning,
            "scores": evaluate(r, seen_ext),
        }
    report["total_seconds"] = round(time.perf_counter() - started, 1)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"rows: {report['rows']}  (external rows mentioning a train merchant: {len(seen_ext)})")
    print(
        "| Model | val macro-F1 | test acc | test macro-F1 | external acc | external macro-F1 "
        "| external unseen macro-F1 |"
    )
    print("|---|---|---|---|---|---|---|")
    for name, m in report["models"].items():
        s = m["scores"]
        u = s.get("external_test_unseen_merchants", {})
        print(
            f"| {name} | {s['val']['macro_f1']:.3f} | {s['test']['accuracy']:.3f} | {s['test']['macro_f1']:.3f} | "
            f"{s['external_test']['accuracy']:.3f} | {s['external_test']['macro_f1']:.3f} | "
            f"{u.get('macro_f1', float('nan')):.3f} |"
        )
    print(
        f"\nsaved predictions/*.csv and {Path(METRICS_PATH).relative_to(REPO_ROOT)} "
        f"(commit {report['commit']}, {report['total_seconds']}s)"
    )
    return 0
