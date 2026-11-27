"""Error analysis (SPEC §7, M6): group the best model's mistakes into error types, and show which model fixes which.

The best model on the independent set is E5 (few-shot LLM; M5). Its external-test errors are sorted into 6 types by
deterministic rules, applied in priority order, so the grouping is reproducible and not hand-picked. The rules were
written after reading a seeded random sample of 60 E5 errors. That is fine for *describing* errors (no model,
prompt or hyperparameter is changed because of them); it would not be fine for tuning.

For each type we report how many rows of that type every other model gets right ("which model fixes which error").
The SPEC's "top 30 misclassified rows" are a seeded random sample of 30 errors (the LLM gives no confidence score to
rank by), listed with their assigned type.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 0
TOP_N = 30

# Payment rails and wrappers: text added by the bank/processor around (or instead of) the merchant name.
RAIL = re.compile(
    r"PAYPAL|PYPL|\bPP\*|\bSQ \*|\bTST\*|WITHDRAWAL|PREAUTHORIZED|PREAPPROVED|INST XFER|WEB ID|PPD ID|BILL ?PAY"
    r"|ONLINE PMT|EBILL|\bACH\b|AUTOPAY",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ErrorType:
    key: str
    title: str
    meaning: str


ERROR_TYPES: list[ErrorType] = [
    ErrorType(
        "invalid_reply", "Invalid reply", "The LLM answered with a non-category such as 'Unknown' or 'Transfer'."
    ),
    ErrorType(
        "credit_on_expense",
        "Credit on an expense category",
        "Money coming IN on an expense category (a refund or reversal, e.g. a hotel refund) read as Income.",
    ),
    ErrorType(
        "taxonomy_boundary",
        "Taxonomy boundary (true 'Other')",
        "The source dataset files groceries, pharmacy, medical, ATM cash etc. under Other (personal spending); the "
        "model picks the nearest business category (e.g. groceries → Meals).",
    ),
    ErrorType(
        "rail_wrapper",
        "Misread payment-rail text",
        "Processor or bank wording ('WITHDRAWAL FROM', 'PAYPAL INST XFER', 'WEB ID') around the merchant drives the "
        "answer, e.g. 'WITHDRAWAL FROM PG AND E' → Bank Fees.",
    ),
    ErrorType(
        "hedge_other",
        "Hedged to 'Other'",
        "A recognisable merchant of a specific category (restaurant, cruise line, utility) answered as Other.",
    ),
    ErrorType("confusion", "Category confusion", "Any other wrong specific category (e.g. car rental → Rent & Lease)."),
]
TYPE_KEYS = [t.key for t in ERROR_TYPES]


def classify_error(description: str, direction: str | None, true: str, pred: str, categories: Iterable[str]) -> str:
    """Error type of one misclassified row (priority order = ERROR_TYPES order)."""
    if true == pred:
        raise ValueError("not an error")
    if pred not in set(categories):
        return "invalid_reply"
    if direction == "credit" and true not in ("Income / Deposits", "Other"):
        return "credit_on_expense"
    if true == "Other":
        return "taxonomy_boundary"
    if pred != "Other" and RAIL.search(description or ""):
        return "rail_wrapper"
    if pred == "Other":
        return "hedge_other"
    return "confusion"


def analyze(rows: pd.DataFrame, preds: dict[str, np.ndarray], best: str, categories: list[str]) -> dict:
    """rows: id, description, direction, category (aligned with each array in preds). Returns counts, fix rates and
    the sample of TOP_N errors."""
    rows = rows.reset_index(drop=True)
    y = rows["category"].to_numpy()
    p_best = preds[best]
    wrong = p_best != y
    err = rows[wrong].copy()
    err["pred"] = p_best[wrong]
    err["type"] = [classify_error(r.description, r.direction, r.category, r.pred, categories) for r in err.itertuples()]
    counts = err["type"].value_counts().reindex(TYPE_KEYS, fill_value=0)

    fixes: dict[str, dict[str, float]] = {}
    idx = np.flatnonzero(wrong)
    for key in TYPE_KEYS:
        rows_k = idx[(err["type"] == key).to_numpy()]
        fixes[key] = {
            m: (float(np.mean(p[rows_k] == y[rows_k])) if len(rows_k) else float("nan"))
            for m, p in preds.items()
            if m != best
        }

    sample = err.sample(n=min(TOP_N, len(err)), random_state=SEED).sort_values(["type", "category"])
    for m, p in preds.items():
        if m != best:
            sample[f"{m}_right"] = p[sample.index.to_numpy()] == sample["category"].to_numpy()
    return {
        "best": best,
        "n_rows": int(len(y)),
        "n_errors": int(wrong.sum()),
        "counts": {k: int(v) for k, v in counts.items()},
        "by_true_class": {
            k: {str(c): int(n) for c, n in err[err["type"] == k]["category"].value_counts().items()} for k in TYPE_KEYS
        },
        "fixed_by": fixes,
        "sample": sample,
    }
