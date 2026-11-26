"""Dataset summary: per-category × split counts (markdown) and a class-distribution chart."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed (CI, servers)
import matplotlib.pyplot as plt  # noqa: E402

from txncat.data.taxonomy import category_names  # noqa: E402

SPLITS = ("train", "val", "test")


def counts_from_csv(path: Path) -> tuple[Counter[tuple[str, str]], Counter[tuple[str, str]]]:
    """(category, split) → row count, and (category, split) → distinct merchant count."""
    rows: Counter[tuple[str, str]] = Counter()
    merchants: dict[tuple[str, str], set[str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["category"], r["split"])
            rows[key] += 1
            merchants.setdefault(key, set()).add(r["merchant"])
    return rows, Counter({k: len(v) for k, v in merchants.items()})


def markdown_table(rows: Counter[tuple[str, str]], merchants: Counter[tuple[str, str]]) -> str:
    lines = [
        "| Category | " + " | ".join(f"{s} rows (merchants)" for s in SPLITS) + " | total | share |",
        "|---|" + "---|" * (len(SPLITS) + 2),
    ]
    total = sum(rows.values())
    for cat in category_names():
        cells = [f"{rows[(cat, s)]} ({merchants[(cat, s)]})" for s in SPLITS]
        n = sum(rows[(cat, s)] for s in SPLITS)
        lines.append(f"| {cat} | " + " | ".join(cells) + f" | {n} | {n / total:.1%} |")
    split_totals = [sum(v for (_, s), v in rows.items() if s == sp) for sp in SPLITS]
    lines.append(
        "| **All** | " + " | ".join(f"**{n}** ({n / total:.1%})" for n in split_totals) + f" | **{total}** | 100% |"
    )
    return "\n".join(lines)


def class_distribution_chart(rows: Counter[tuple[str, str]], out: Path) -> Path:
    cats = category_names()
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    left = [0] * len(cats)
    colors = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
    for split in SPLITS:
        vals = [rows[(c, split)] for c in cats]
        ax.barh(cats, vals, left=left, label=split, color=colors[split])
        left = [a + b for a, b in zip(left, vals, strict=True)]
    ax.invert_yaxis()
    ax.set_xlabel("transactions")
    ax.set_title(f"Synthetic transactions by category and split (n={sum(rows.values()):,})")
    ax.legend(title="split (by merchant)", loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
