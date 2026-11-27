"""`txncat report`: every M5 table and figure, regenerated from saved predictions and metrics files only.

Inputs : predictions/*.csv (one row per transaction per split), reports/metrics/*.json (fit times, LLM latency),
         reports/metrics/latency.json (from `txncat bench latency`; optional).
Outputs: reports/results_tables.md, reports/metrics/significance.json, reports/figures/{f1_ci,latency_vs_f1,
         confusion_*}.png. No database, no model, no network: anyone with the repo can rerun it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from txncat.config import REPO_ROOT
from txncat.eval.significance import N_BOOT, bootstrap_diff, bootstrap_macro_f1, holm, mcnemar

PRED_DIR = REPO_ROOT / "predictions"
METRICS_DIR = REPO_ROOT / "reports" / "metrics"
FIG_DIR = REPO_ROOT / "reports" / "figures"
TABLES_MD = REPO_ROOT / "reports" / "results_tables.md"
SPLITS = ("test", "external_test")
ALPHA = 0.05

# name -> (label in tables, prediction files). E3 has one file per seed.
MODELS: dict[str, tuple[str, list[str]]] = {
    "E0_rules": ("E0 keyword rules", ["E0_rules.csv"]),
    "E1": ("E1 logreg TF-IDF", ["E1_logreg_tfidf.csv"]),
    "E2": ("E2 XGBoost TF-IDF+SQL", ["E2_xgb_tfidf_sql.csv"]),
    "E3": ("E3 DistilBERT", [f"E3_finetune_seed{s}.csv" for s in (0, 1, 2)]),
    "E4": ("E4 LLM zero-shot", ["E4_llm_zero_shot.csv"]),
    "E5": ("E5 LLM few-shot", ["E5_llm_few_shot.csv"]),
}
# SPEC §7 comparisons, plus E4 vs E5 (does few-shot help?).
PAIRS: list[tuple[str, str]] = [("E2", "E3"), ("E2", "E5"), ("E3", "E5"), ("E4", "E5")]


@dataclass
class Preds:
    """Aligned predictions for one split: y and one prediction column per (model, variant)."""

    split: str
    ids: np.ndarray
    y: np.ndarray
    cols: dict[str, list[np.ndarray]]  # model -> one array per seed/variant


def load_split(split: str, pred_dir: Path = PRED_DIR) -> Preds:
    base: pd.DataFrame | None = None
    cols: dict[str, list[np.ndarray]] = {}
    frames: dict[str, list[pd.DataFrame]] = {}
    for model, (_, files) in MODELS.items():
        parts = []
        for f in files:
            path = pred_dir / f
            if not path.exists():
                break
            d = pd.read_csv(path, usecols=["id", "split", "category", "pred"], keep_default_na=False)
            parts.append(d[d["split"] == split].set_index("id").sort_index())
        if parts and len(parts) == len(files) and len(parts[0]):
            frames[model] = parts
    if not frames:
        raise FileNotFoundError(f"no predictions for split {split!r} in {pred_dir}")
    ids = None
    for parts in frames.values():
        for p in parts:
            if ids is None:
                ids, base = p.index.to_numpy(), p
            elif not np.array_equal(p.index.to_numpy(), ids):
                raise ValueError(f"prediction files disagree on the {split} rows; rerun the models")
            elif not (p["category"].to_numpy() == base["category"].to_numpy()).all():  # type: ignore[index]
                raise ValueError(f"prediction files disagree on the {split} labels")
    for model, parts in frames.items():
        cols[model] = [p["pred"].astype(str).to_numpy() for p in parts]
    assert base is not None and ids is not None
    return Preds(split, ids, base["category"].astype(str).to_numpy(), cols)


def score_table(p: Preds, n_boot: int = N_BOOT) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model, variants in p.cols.items():
        rows = []
        for pred in variants:
            f1, lo, hi = bootstrap_macro_f1(p.y, pred, n_boot=n_boot)
            rows.append({"macro_f1": f1, "ci95": [lo, hi], "accuracy": float(np.mean(pred == p.y))})
        entry: dict[str, Any] = {"n": len(p.y), "variants": rows}
        if len(rows) > 1:
            f1s = np.array([r["macro_f1"] for r in rows])
            entry["mean_macro_f1"], entry["std_macro_f1"] = float(f1s.mean()), float(f1s.std(ddof=1))
        out[model] = entry
    return out


def significance(p: Preds, n_boot: int = N_BOOT) -> list[dict[str, Any]]:
    """One McNemar test per (pair, seed variant); Holm-adjusted across everything tested on this split.
    A pair involving E3 counts as significant only if every seed is significant in the same direction."""
    tests: list[dict[str, Any]] = []
    for a, b in PAIRS:
        if a not in p.cols or b not in p.cols:
            continue
        for ia, pa in enumerate(p.cols[a]):
            for ib, pb in enumerate(p.cols[b]):
                m = mcnemar(p.y, pa, pb)
                d, lo, hi = bootstrap_diff(p.y, pa, pb, n_boot=n_boot)
                tests.append(
                    {
                        "a": a,
                        "b": b,
                        "a_variant": ia,
                        "b_variant": ib,
                        **m.as_dict(),
                        "macro_f1_diff": d,
                        "diff_ci95": [lo, hi],
                    }
                )
    for t, adj in zip(tests, holm([t["p_value"] for t in tests]), strict=True):
        t["p_holm"] = adj
    pairs: list[dict[str, Any]] = []
    for a, b in PAIRS:
        group = [t for t in tests if t["a"] == a and t["b"] == b]
        if not group:
            continue
        signs = {np.sign(t["macro_f1_diff"]) for t in group}
        sig = all(t["p_holm"] < ALPHA for t in group) and len(signs) == 1
        better = (a if group[0]["macro_f1_diff"] > 0 else b) if sig else None
        pairs.append(
            {
                "a": a,
                "b": b,
                "tests": group,
                "significant": sig,
                "better": better,
                "max_p_holm": max(t["p_holm"] for t in group),
            }
        )
    return pairs


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def cost_latency(metrics_dir: Path = METRICS_DIR) -> dict[str, dict[str, Any]]:
    """API cost (all $0: nothing calls a paid API), training compute and per-prediction latency."""
    base = _read_json(metrics_dir / "baseline.json").get("models", {})
    e3 = _read_json(metrics_dir / "e3_finetune.json")
    lat = _read_json(metrics_dir / "latency.json").get("models", {})
    out: dict[str, dict[str, Any]] = {}
    for key, name in (("E1", "E1_logreg_tfidf"), ("E2", "E2_xgb_tfidf_sql")):
        out[key] = {
            "cost_usd_per_1000": 0.0,
            "train_seconds": base.get(name, {}).get("fit_seconds"),
            "train_note": "grid search on val, all candidates",
            "latency": lat.get(name),
        }
    seeds = e3.get("seeds", [])
    out["E3"] = {
        "cost_usd_per_1000": 0.0,
        "train_seconds": float(np.mean([s["seconds"] for s in seeds])) if seeds else None,
        "train_note": f"per seed, {seeds[0]['device'] if seeds else '?'}",
        "latency": lat.get("E3_finetune"),
        "latency_gpu": lat.get("E3_finetune_gpu"),
    }
    for key, f in (("E4", "E4_llm_zero_shot.json"), ("E5", "E5_llm_few_shot.json")):
        m = _read_json(metrics_dir / f)
        lm = m.get("latency_ms", {})
        out[key] = {
            "cost_usd_per_1000": m.get("cost_usd_per_1000", 0.0),
            "train_seconds": 0.0,
            "train_note": "no training",
            "latency": {
                "device": m.get("gpu") or m.get("device"),
                "p50_ms": lm.get("p50"),
                "p95_ms": lm.get("p95"),
                "mean_ms": lm.get("mean"),
            }
            if lm
            else None,
        }
    return out


# ----------------------------------------------------------------------------------------------------- markdown
def _f(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _ms(lat: dict[str, Any] | None) -> str:
    if not lat or lat.get("p50_ms") is None:
        return "not measured"
    return f"{lat['p50_ms']:.2f} / {lat['p95_ms']:.2f} ({lat['device']})"


def markdown(scores: dict[str, dict], sig: dict[str, list], cost: dict[str, dict], meta: dict) -> str:
    L = [
        "# Results tables (M5)",
        "",
        f"Generated by `{meta['command']}` on {meta['date']} at commit `{meta['commit']}`, "
        f"from `predictions/` and `reports/metrics/` only. Bootstrap: {meta['n_boot']} row resamples, percentile 95% "
        "CI. Macro-F1 averages the categories present in each split (13 on test, 7 on external); invalid LLM "
        "replies count as wrong.",
        "",
    ]
    L += [
        "## Macro-F1 with 95% bootstrap CIs",
        "",
        "| Model | test macro-F1 [95% CI] | test acc | external macro-F1 [95% CI] | external acc |",
        "|---|---|---|---|---|",
    ]
    for model, (label, _) in MODELS.items():
        cells = []
        for split in SPLITS:
            e = scores.get(split, {}).get(model)
            if e is None:
                cells += ["n/a", "n/a"]
                continue
            v = e["variants"]
            if len(v) == 1:
                cells += [
                    f"{v[0]['macro_f1']:.3f} [{v[0]['ci95'][0]:.3f}, {v[0]['ci95'][1]:.3f}]",
                    f"{v[0]['accuracy']:.3f}",
                ]
            else:
                lo = min(r["ci95"][0] for r in v)
                hi = max(r["ci95"][1] for r in v)
                cells += [
                    f"{e['mean_macro_f1']:.3f} ± {e['std_macro_f1']:.3f} (seed CIs span [{lo:.3f}, {hi:.3f}])",
                    f"{np.mean([r['accuracy'] for r in v]):.3f}",
                ]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L += ["", "E3 = mean ± std over 3 seeds; per-seed CIs are in `reports/metrics/significance.json`.", ""]
    L += [
        "## Pairwise significance (McNemar, exact, Holm-corrected per split)",
        "",
        "Differences are macro-F1(A) − macro-F1(B) with a paired bootstrap 95% CI. For pairs with E3, each seed is "
        "tested separately and the pair counts as significant only if all 3 seeds are (same direction); the row "
        "shows the range over seeds and the largest Holm p.",
        "",
        "McNemar compares error *rates* (which model gets more rows right); the bootstrap CI compares *macro-F1*. "
        "They can disagree when one model is better on rare classes and the other on common ones.",
        "",
        "| Split | A vs B | discordant rows: A right & B wrong / A wrong & B right | ΔF1 (A−B) [95% CI] "
        "| Holm p (max) | McNemar significant at 0.05? |",
        "|---|---|---|---|---|---|",
    ]
    for split in SPLITS:
        for pr in sig.get(split, []):
            t = pr["tests"]
            disc = ", ".join(f"{x['a_right_b_wrong']} / {x['a_wrong_b_right']}" for x in t)
            if len(t) > 1:
                disc += " (per E3 seed 0, 1, 2)"
            if len(t) == 1:
                d = f"{t[0]['macro_f1_diff']:+.3f} [{t[0]['diff_ci95'][0]:+.3f}, {t[0]['diff_ci95'][1]:+.3f}]"
            else:
                ds = [x["macro_f1_diff"] for x in t]
                d = f"{min(ds):+.3f} to {max(ds):+.3f} over seeds"
            p = pr["max_p_holm"]
            verdict = f"yes, {pr['better']} better" if pr["significant"] else "no"
            L.append(f"| {split} | {pr['a']} vs {pr['b']} | {disc} | {d} | {p:.2g} | {verdict} |")
    L += [
        "",
        "## Cost and latency",
        "",
        "All models run locally; no paid API is called anywhere, so API cost is $0 for every model. Latency is one "
        "transaction per call (batch size 1), p50 / p95 in milliseconds, on the device shown.",
        "",
        "| Model | API cost per 1,000 | training compute | latency p50 / p95 ms |",
        "|---|---|---|---|",
    ]
    for key in ("E1", "E2", "E3", "E4", "E5"):
        c = cost.get(key, {})
        tr = (
            "none"
            if c.get("train_seconds") == 0.0
            else (f"{c['train_seconds']:.0f} s ({c['train_note']})" if c.get("train_seconds") is not None else "n/a")
        )
        lat = _ms(c.get("latency"))
        if c.get("latency_gpu"):
            lat += f"; {_ms(c['latency_gpu'])}"
        L.append(f"| {MODELS[key][0]} | ${c.get('cost_usd_per_1000', 0.0):.2f} | {tr} | {lat} |")
    L += [
        "",
        "E3 latency is measured on the base DistilBERT architecture with a 13-way head (identical compute to the "
        "fine-tuned weights). E4/E5 latency was recorded while generating their predictions.",
        "",
    ]
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------------ figures
def figures(
    scores: dict[str, dict], cost: dict[str, dict], preds: dict[str, Preds], fig_dir: Path = FIG_DIR
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from txncat.data.taxonomy import category_names
    from txncat.eval.metrics import confusion

    fig_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    blue, orange, grey = "#2a6fdb", "#e07b39", "#6b7280"

    def point(split: str, model: str) -> tuple[float, float, float] | None:
        e = scores.get(split, {}).get(model)
        if e is None:
            return None
        v = e["variants"]
        if len(v) == 1:
            return v[0]["macro_f1"], v[0]["ci95"][0], v[0]["ci95"][1]
        return e["mean_macro_f1"], min(r["ci95"][0] for r in v), max(r["ci95"][1] for r in v)

    # 1. macro-F1 with CIs, test vs external
    models = list(MODELS)
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for off, split, color, name in (
        (-0.2, "test", blue, "test (unseen synthetic merchants)"),
        (0.2, "external_test", orange, "external (independent dataset)"),
    ):
        pts = [point(split, m) for m in models]
        vals = [p[0] if p else np.nan for p in pts]
        err = np.array([[p[0] - p[1], p[2] - p[0]] if p else [0, 0] for p in pts]).T
        ax.bar(x + off, vals, 0.4, color=color, label=name, yerr=err, capsize=3, error_kw={"elinewidth": 1})
        for xi, v in zip(x + off, vals, strict=True):
            if not np.isnan(v):
                ax.text(xi, 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="white")
    ax.set_xticks(x, [MODELS[m][0].replace(" ", "\n", 1) for m in models], fontsize=8)
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title("Macro-F1 with 95% bootstrap CIs (E3: mean of 3 seeds, CI span over seeds)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.append(fig_dir / "f1_ci.png")
    fig.savefig(out[-1], dpi=150)
    plt.close(fig)

    # 2. latency vs external macro-F1 (cost is $0 for everything, so latency is the trade-off axis)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key in ("E1", "E2", "E3", "E4", "E5"):
        lat = (cost.get(key) or {}).get("latency")
        if not lat or lat.get("p50_ms") is None:
            continue
        for split, color, marker in (("test", blue, "o"), ("external_test", orange, "s")):
            p = point(split, key)
            if p:
                ax.scatter(lat["p50_ms"], p[0], color=color, marker=marker, s=40, zorder=3)
        p = point("external_test", key)
        if p:
            ax.annotate(key, (lat["p50_ms"], p[0]), textcoords="offset points", xytext=(6, -10), fontsize=8, color=grey)
    ax.scatter([], [], color=blue, marker="o", label="test")
    ax.scatter([], [], color=orange, marker="s", label="external")
    ax.set_xscale("log")
    ax.set_xlabel("p50 latency per prediction, ms (log scale; E1–E3 CPU, E4/E5 laptop GPU)")
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy vs latency (API cost is $0 for all models)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.append(fig_dir / "latency_vs_f1.png")
    fig.savefig(out[-1], dpi=150)
    plt.close(fig)

    # 3. confusion matrices: best test model and best external model
    labels = category_names()
    for split, model in (("test", "E2"), ("external_test", "E5"), ("external_test", "E2")):
        sp = preds.get(split)
        if sp is None or model not in sp.cols:
            continue
        present = [lab for lab in labels if np.any(sp.y == lab)]  # rows: true classes in this split
        cm = confusion(sp.y, sp.cols[model][0], labels)[[labels.index(lab) for lab in present]]
        # divide by the true count, not the row sum: INVALID LLM replies are off the 13 columns but still errors
        row = np.array([[np.sum(sp.y == lab)] for lab in present])
        norm = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row > 0)
        fig, ax = plt.subplots(figsize=(7.5, 2.0 + 0.4 * len(present)))
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(present)), present, fontsize=7)
        for i in range(len(present)):
            for j in range(len(labels)):
                if cm[i, j]:
                    ax.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black",
                    )
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(
            f"{MODELS[model][0]} on {split}\n(counts; colour = share of the true class; rows can sum short of the "
            "class size where the LLM reply was invalid)",
            fontsize=8,
        )
        fig.tight_layout()
        out.append(fig_dir / f"confusion_{model}_{split}.png")
        fig.savefig(out[-1], dpi=150)
        plt.close(fig)
    return out


def run(n_boot: int = N_BOOT, make_figures: bool = True) -> int:
    from txncat.models.train_cli import _git_commit

    meta = {
        "command": "txncat report",
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "n_boot": n_boot,
        "alpha": ALPHA,
    }
    preds = {s: load_split(s, PRED_DIR) for s in SPLITS}
    scores = {s: score_table(p, n_boot) for s, p in preds.items()}
    sig = {s: significance(p, n_boot) for s, p in preds.items()}
    cost = cost_latency(METRICS_DIR)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "significance.json").write_text(
        json.dumps({**meta, "scores": scores, "pairs": sig, "cost_latency": cost}, indent=2), encoding="utf-8"
    )
    TABLES_MD.write_text(markdown(scores, sig, cost, meta), encoding="utf-8")
    print(TABLES_MD.read_text(encoding="utf-8"))
    paths = figures(scores, cost, preds, FIG_DIR) if make_figures else []
    for path in [METRICS_DIR / "significance.json", TABLES_MD, *paths]:
        print(f"saved {path.relative_to(REPO_ROOT)}")
    return 0
