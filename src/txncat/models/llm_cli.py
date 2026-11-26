"""`txncat llm run`: E4/E5 with a free local open-weight model; saves predictions and metrics (no paid API)."""

from __future__ import annotations

import json
import statistics
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from txncat.config import REPO_ROOT
from txncat.eval.metrics import score
from txncat.models.llm_fewshot import INVALID, run_llm
from txncat.models.train_cli import PRED_DIR, _git_commit, external_seen_ids

NAMES = {"zero": "E4_llm_zero_shot", "few": "E5_llm_few_shot"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_rows() -> pd.DataFrame:
    from txncat.db import apply_schema, make_engine

    engine = make_engine()
    apply_schema(engine)
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT id, split, category, description, amount, direction FROM transactions ORDER BY id"), conn
        )
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df


def run(mode: str, splits: list[str], limit: int | None, model_id: str, threads: int | None) -> int:
    from txncat.models.llm_client import LocalHFClient, ResponseCache

    df = load_rows()
    client = LocalHFClient(model_id=model_id, cache=ResponseCache(), threads=threads)
    log(f"model {client.model_id}@{(client.revision or '')[:8]} loaded; mode={mode}")
    r = run_llm(df, client, mode, tuple(splits), limit_per_split=limit, log=log)

    name = NAMES[mode] + (f"_sample{limit}" if limit else "")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    r.predictions.to_csv(PRED_DIR / f"{name}.csv", index=False)

    seen = external_seen_ids(df)
    p = r.predictions
    scores: dict[str, Any] = {}
    for split in splits:
        part = p[p["split"] == split]
        s = score(part["category"], part["pred"]).as_dict()
        s["invalid"] = int((part["pred"] == INVALID).sum())
        scores[split] = s
        if split == "external_test" and seen:
            unseen = part[~part["id"].isin(seen)]
            if len(unseen):
                scores["external_test_unseen_merchants"] = score(unseen["category"], unseen["pred"]).as_dict()
    lat = p["latency_ms"].to_numpy()
    report = {
        "command": f"txncat llm run --mode {mode} --splits {' '.join(splits)}"
        + (f" --limit {limit}" if limit else "")
        + f" --model {model_id}",
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "model": client.model_id,
        "revision": client.revision,
        "license": "apache-2.0",
        "device": client.device,
        "gpu": _gpu_name(client.device),
        "decoding": client.settings,
        "prompt_sha256": r.prompt_sha256,
        "few_shot_example_ids": r.example_ids,
        "rows": int(len(p)),
        "cache_hits": int(p["cached"].sum()),
        "latency_ms": {
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
            "mean": float(lat.mean()),
            "note": "per prediction on the recorded device, measured when first generated (cache hits keep it)",
        },
        "cost_usd_per_1000": 0.0,
        "cpu_seconds_per_1000": float(lat.mean()),  # mean ms per prediction == CPU-seconds per 1000 predictions
        "wall_seconds_this_run": round(r.seconds, 1),
        "scores": scores,
    }
    out = REPO_ROOT / "reports" / "metrics" / f"{name}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"| {name} | macro-F1 | accuracy | invalid replies |")
    print("|---|---|---|---|")
    for split, s in scores.items():
        print(f"| {split} | {s['macro_f1']:.3f} | {s['accuracy']:.3f} | {s.get('invalid', '')} |")
    print(f"latency p50 {statistics.median(lat):.0f} ms, p95 {np.percentile(lat, 95):.0f} ms; cost $0 (local model)")
    print(f"saved predictions/{name}.csv and {out.relative_to(REPO_ROOT)}")
    return 0


def _gpu_name(device: str) -> str | None:
    if device != "cuda":
        return None
    import torch

    return str(torch.cuda.get_device_name(0))
