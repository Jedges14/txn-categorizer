"""External test set: an independent, MIT-licensed transaction dataset, mapped onto our taxonomy.

Source: DoDataThings/us-bank-transaction-categories-v2 (Hugging Face), pinned to one revision and checked by SHA-256.
It is SYNTHETIC too, but from someone else's generator with different formatting (mixed case, untruncated,
ACH fields), so it tests whether models generalize beyond our generator's quirks. It is not a "real" test set.

Only categories with an unambiguous counterpart are kept; the rest are excluded (see EXCLUDED). The file has no
amounts or dates; those stay NULL. Its "[debit]"/"[credit]" marker becomes the `direction` column.
"""

from __future__ import annotations

import csv
import hashlib
import random
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

from txncat.config import REPO_ROOT

REPO_ID = "DoDataThings/us-bank-transaction-categories-v2"
REVISION = "3e8d052cb7b362fb1a04b6064d2331982553ea56"
FILENAME = "transactions-synthetic.csv"
SHA256 = "0424aed6e76f74a5b3b1ff61ccec43bc321622e6806da353b910b3b2c8108f6e"
URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{FILENAME}"
LOCAL_PATH = REPO_ROOT / "data" / "external" / FILENAME
SOURCE = "external:dodatathings-v2"

CATEGORY_MAP: dict[str, str] = {
    "Restaurants": "Meals & Entertainment",
    "Travel": "Travel",
    "Utilities": "Utilities",
    "Rent": "Rent & Lease",
    "Fees": "Bank Fees & Charges",
    "Income": "Income / Deposits",
    "Transfer": "Other",
    "Groceries": "Other",
    "Healthcare": "Other",
    "Personal Care": "Other",
}
# Excluded, with the reason (documented in data/README.md):
EXCLUDED: dict[str, str] = {
    "Subscription": "mixes business SaaS (ClickUp) with consumer apps (Instagram, Peloton)",
    "Shopping": "could be Office Supplies or Other",
    "Transportation": "could be Fuel & Auto or Travel",
    "Entertainment": "could be Meals & Entertainment or Other (streaming)",
    "Insurance": "no matching category",
    "Mortgage": "no matching category",
    "Education": "no matching category",
}
_DIRECTION = re.compile(r"^\[(debit|credit)\]\s*", re.IGNORECASE)


class ExternalDataError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: Path = LOCAL_PATH, url: str = URL, expected_sha256: str = SHA256) -> Path:
    """Download the pinned file once; refuse to use it if the checksum doesn't match."""
    if dest.exists() and sha256_of(dest) == expected_sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "txn-categorizer/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:  # noqa: S310 (fixed https URL)
        out.write(resp.read())
    actual = sha256_of(tmp)
    if actual != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise ExternalDataError(f"checksum mismatch for {url}: expected {expected_sha256}, got {actual}")
    tmp.replace(dest)
    return dest


def overlap_with_merchants(rows: list[dict], merchant_names: set[str], min_len: int = 4) -> list[str | None]:
    """For each row, the (longest) known merchant name it mentions, or None.

    Used to report external-test scores separately for rows whose merchant also appears in training data.
    Names shorter than `min_len` (e.g. "76", "BP") are skipped to avoid accidental substring matches.
    """
    names = sorted((n for n in merchant_names if len(n) >= min_len), key=len, reverse=True)
    if not names:
        return [None] * len(rows)
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b")
    return [(m.group(1) if (m := pattern.search(r["description"].upper())) else None) for r in rows]


def build_external_test(path: Path = LOCAL_PATH, per_category: int = 300, seed: int = 42) -> list[dict]:
    """A fixed, seeded sample of `per_category` rows for each mapped target category."""
    if sha256_of(path) != SHA256 and path == LOCAL_PATH:
        raise ExternalDataError(f"{path} does not match the pinned checksum; run `txncat data external`")
    by_target: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            target = CATEGORY_MAP.get(r["category"])
            if target is None:
                continue
            raw = r["description"].strip()
            m = _DIRECTION.match(raw)
            direction = m.group(1).lower() if m else None
            desc = _DIRECTION.sub("", raw).strip()
            if not desc:
                continue
            by_target[target][r["category"]].append(
                {
                    "posted_at": None,
                    "description": desc,
                    "amount": None,
                    "direction": direction,
                    "category": target,
                    "split": "external_test",
                    "source": SOURCE,
                }
            )
    rng = random.Random(seed)
    rows: list[dict] = []
    for target in sorted(by_target):
        sources = sorted(by_target[target])
        # Spread each target's quota evenly over its source categories (e.g. Other ← 4 sources × 75).
        base, extra = divmod(per_category, len(sources))
        for i, src in enumerate(sources):
            unique = {r["description"]: r for r in by_target[target][src]}  # drop exact duplicate strings
            pool = [unique[d] for d in sorted(unique)]
            rows.extend(rng.sample(pool, min(len(pool), base + (1 if i < extra else 0))))
    return rows
