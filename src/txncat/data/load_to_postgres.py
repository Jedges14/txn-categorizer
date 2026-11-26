"""Load synthetic rows, the external test set and (if present) the private real test set into PostgreSQL.

Loading is a full replace per source ('synthetic', 'external:…', 'real'), so reruns don't duplicate rows.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from sqlalchemy import Engine, text

from txncat.data.taxonomy import category_names

REAL_TEST_COLUMNS = ["posted_at", "description", "amount", "category"]
INSERT = text(
    "INSERT INTO transactions (posted_at, description, amount, direction, category, split, source) "
    "VALUES (:posted_at, :description, :amount, :direction, :category, :split, :source)"
)


class RealTestError(ValueError):
    """The real test CSV doesn't match the template; message lists the bad rows."""


@dataclass(frozen=True)
class LoadSummary:
    source: str
    rows: int
    by_split: dict[str, int]


def read_synthetic_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return [
            {
                "posted_at": datetime.fromisoformat(r["posted_at"]),
                "description": r["description"],
                "amount": Decimal(r["amount"]),
                "category": r["category"],
                "split": r["split"],
                "source": "synthetic",
            }
            for r in csv.DictReader(f)
        ]


def read_real_test_csv(path: Path) -> list[dict]:
    """Validate the hand-labeled file (template: data/sample/real_test_example.csv) and return rows."""
    allowed = set(category_names())
    errors: list[str] = []
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or [c.strip() for c in reader.fieldnames] != REAL_TEST_COLUMNS:
            raise RealTestError(f"header must be exactly: {','.join(REAL_TEST_COLUMNS)}")
        for i, r in enumerate(reader, start=2):  # line 1 is the header
            try:
                posted = datetime.fromisoformat(r["posted_at"].strip())
            except ValueError:
                errors.append(f"line {i}: posted_at must be YYYY-MM-DD, got {r['posted_at']!r}")
                continue
            try:
                amount = Decimal(r["amount"].strip())
            except InvalidOperation:
                errors.append(f"line {i}: amount must be a number like -12.34, got {r['amount']!r}")
                continue
            category = r["category"].strip()
            if category not in allowed:
                errors.append(f"line {i}: unknown category {category!r} (see data/taxonomy.yaml)")
                continue
            if not r["description"].strip():
                errors.append(f"line {i}: description is empty")
                continue
            rows.append(
                {
                    "posted_at": posted,
                    "description": r["description"].strip(),
                    "amount": amount,
                    "category": category,
                    "split": "real_test",
                    "source": "real",
                }
            )
    if errors:
        raise RealTestError("\n".join(errors[:20]) + (f"\n… and {len(errors) - 20} more" if len(errors) > 20 else ""))
    return rows


def replace_source(engine: Engine, source: str, rows: list[dict]) -> LoadSummary:
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM transactions WHERE source = :s"), {"s": source})
        if rows:
            conn.execute(INSERT, [{"direction": None, **r} for r in rows])
        counts = conn.execute(
            text("SELECT split, count(*) FROM transactions WHERE source = :s GROUP BY split"),
            {"s": source},
        ).all()
    return LoadSummary(source, len(rows), {str(s): int(n) for s, n in counts})
