"""`txncat train finetune` (E3) and `txncat data export-encoder` (input file for Colab)."""

from __future__ import annotations

import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from txncat.config import REPO_ROOT
from txncat.eval.metrics import score
from txncat.models.train_cli import PRED_DIR, _git_commit, external_seen_ids

ENCODER_CSV = REPO_ROOT / "data" / "generated" / "encoder_input.csv"
METRICS_PATH = REPO_ROOT / "reports" / "metrics" / "e3_finetune.json"
COLUMNS = ["id", "split", "category", "description", "desc_clean"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_frame(input_csv: Path | None) -> pd.DataFrame:
    if input_csv is not None:
        return pd.read_csv(input_csv)
    from txncat.db import apply_schema, make_engine
    from txncat.features.sql_features import load_features

    engine = make_engine()
    apply_schema(engine)
    return load_features(engine)[COLUMNS]


def export_encoder_input(path: Path = ENCODER_CSV) -> Path:
    df = load_frame(None)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _mean_std(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def run(seeds: list[int], input_csv: Path | None, epochs: int) -> int:
    from txncat.models.finetune import FinetuneConfig, config_dict, train_one_seed

    df = load_frame(input_csv)
    seen = external_seen_ids(df)
    per_seed: list[dict[str, Any]] = []
    cfg = FinetuneConfig(epochs=epochs)
    for seed in seeds:
        cfg = FinetuneConfig(seed=seed, epochs=epochs)
        r = train_one_seed(df, cfg, log=log)
        r.predictions.to_csv(PRED_DIR / f"E3_finetune_seed{seed}.csv", index=False)
        p = r.predictions
        scores: dict[str, Any] = {}
        for split in ("val", "test", "external_test"):
            part = p[p["split"] == split]
            scores[split] = score(part["category"], part["pred"]).as_dict()
        ext = p[p["split"] == "external_test"]
        unseen = ext[~ext["id"].isin(seen)]
        if seen and len(unseen):
            scores["external_test_unseen_merchants"] = score(unseen["category"], unseen["pred"]).as_dict()
        per_seed.append(
            {
                "seed": seed,
                "best_epoch": r.best_epoch,
                "val_macro_f1_by_epoch": r.val_macro_f1_by_epoch,
                "seconds": round(r.seconds, 1),
                "device": r.device,
                "scores": scores,
            }
        )
        log(
            f"E3 seed={seed} done: test macro-F1={scores['test']['macro_f1']:.4f} "
            f"external macro-F1={scores['external_test']['macro_f1']:.4f}"
        )

    summary = {
        split: {
            metric: _mean_std([s["scores"][split][metric] for s in per_seed]) for metric in ("macro_f1", "accuracy")
        }
        for split in per_seed[0]["scores"]
    }
    report = {
        "command": f"txncat train finetune --seeds {' '.join(map(str, seeds))} --epochs {epochs}",
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "config": {k: v for k, v in config_dict(cfg).items() if k != "seed"},
        "seeds": per_seed,
        "summary": summary,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("| E3 (mean ± std over seeds) | macro-F1 | accuracy |")
    print("|---|---|---|")
    for split, m in summary.items():
        print(
            f"| {split} | {m['macro_f1']['mean']:.3f} ± {m['macro_f1']['std']:.3f} | "
            f"{m['accuracy']['mean']:.3f} ± {m['accuracy']['std']:.3f} |"
        )
    print(f"\nsaved predictions/E3_finetune_seed*.csv and {METRICS_PATH.relative_to(REPO_ROOT)}")
    return 0
