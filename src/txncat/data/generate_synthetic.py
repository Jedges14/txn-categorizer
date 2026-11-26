"""Synthetic bank transactions that mimic real statement formatting (SPEC §3b).

Rows are drawn per category (deliberately imbalanced), per merchant (skewed popularity), rendered with
bank-style formatting (processor prefixes, store numbers, city/state, ALL CAPS, 22-32 character truncation),
given category-specific log-normal amounts and dates, and assigned to train/val/test BY MERCHANT so no merchant
appears in two splits. Everything is driven by one seed.
"""

from __future__ import annotations

import csv
import random
import string
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from txncat.data.merchants import Merchant

# Share of rows per category: imbalanced on purpose, roughly like a small business's statement.
CATEGORY_WEIGHTS: dict[str, float] = {
    "Meals & Entertainment": 14,
    "Travel": 7,
    "Fuel & Auto": 10,
    "Office Supplies": 9,
    "Software & Subscriptions": 11,
    "Utilities": 6,
    "Rent & Lease": 3,
    "Advertising & Marketing": 6,
    "Professional Services": 6,
    "Shipping & Postage": 6,
    "Bank Fees & Charges": 5,
    "Income / Deposits": 12,
    "Other": 5,
}

# Log-normal parameters (mu, sigma) for the absolute dollar amount; exp(mu) is the median.
AMOUNT_PARAMS: dict[str, tuple[float, float]] = {
    "Meals & Entertainment": (2.8, 0.6),
    "Travel": (5.0, 0.9),
    "Fuel & Auto": (3.9, 0.7),
    "Office Supplies": (3.6, 0.9),
    "Software & Subscriptions": (3.3, 0.9),
    "Utilities": (4.6, 0.6),
    "Rent & Lease": (7.2, 0.6),
    "Advertising & Marketing": (4.7, 1.0),
    "Professional Services": (6.0, 0.9),
    "Shipping & Postage": (2.8, 0.8),
    "Income / Deposits": (6.5, 1.1),
    "Other": (3.5, 1.0),
}
FEE_AMOUNTS = [2.50, 3.00, 5.00, 12.00, 15.00, 25.00, 30.00, 35.00, 45.00]

SPLIT_FRACTIONS = (("train", 0.70), ("val", 0.15), ("test", 0.15))
START, END = date(2024, 1, 1), date(2025, 12, 31)
REFUND_RATE = 0.015  # small share of expense rows are refunds/returns (positive amount, same category)

CITIES = """OAKLAND CA|SAN FRANCISCO CA|AUSTIN TX|DALLAS TX|DENVER CO|PORTLAND OR|SEATTLE WA|BOSTON MA|
CHICAGO IL|MIAMI FL|ATLANTA GA|PHOENIX AZ|NEW YORK NY|BROOKLYN NY|NEWARK NJ|PHILADELPHIA PA|RALEIGH NC|
NASHVILLE TN|COLUMBUS OH|DETROIT MI|MINNEAPOLIS MN|SAN DIEGO CA|LOS ANGELES CA|SALT LAKE CTY UT|
LAS VEGAS NV|TAMPA FL|ORLANDO FL|RICHMOND VA|BALTIMORE MD|PITTSBURGH PA""".replace("\n", "").split("|")
CARD_PREFIXES = [
    "",
    "",
    "",
    "",
    "POS DEBIT ",
    "DEBIT PURCHASE ",
    "CHECKCARD {mmdd} ",
    "PURCHASE {mmdd} ",
]
# Point-of-sale processors used by small local merchants, limited to the businesses that really use them:
# Toast (TST*) is restaurant-only; Clover/Square/Shopify POS are retail and services; PayPal is anywhere.
LOCAL_PREFIXES: dict[str, list[str]] = {
    "Meals & Entertainment": ["SQ *", "SQ *", "TST* ", "TST* ", "CLV*", "PAYPAL *"],
    "Other": ["SQ *", "CLV*", "SP ", "PAYPAL *"],
    "Office Supplies": ["SQ *", "CLV*", "SP ", "PAYPAL *"],
    "Fuel & Auto": ["SQ *", "CLV*", "PAYPAL *"],
    "Shipping & Postage": ["SQ *", "CLV*", "PAYPAL *"],
    "Advertising & Marketing": ["SQ *", "PAYPAL *"],
    "Professional Services": ["SQ *", "PAYPAL *"],
    "Travel": ["SQ *", "PAYPAL *"],
}
SAAS_PROCESSORS = ["", "", "", "PADDLE.NET* ", "FS *", "2CO.COM*", "PAYPAL *"]
ACH_DEBIT = [
    "ACH DEBIT {m}",
    "{m} PAYMENT",
    "{m} AUTOPAY",
    "ONLINE PMT {m}",
    "{m} WEB PMT",
    "{m} BILL PAY",
]
ACH_CREDIT = [
    "ACH CREDIT {m}",
    "DEPOSIT {m} PAYMENT",
    "{m} ACH PMT",
    "ZELLE FROM {m}",
    "WIRE IN {m}",
]


@dataclass(frozen=True)
class Txn:
    posted_at: datetime
    description: str
    amount: float
    category: str
    merchant: str
    split: str


def _code(rng: random.Random, n: int, alphabet: str = string.ascii_uppercase + string.digits) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


def _digits(rng: random.Random, n: int) -> str:
    return _code(rng, n, string.digits)


def _truncate(s: str, rng: random.Random) -> str:
    """Banks cut descriptors at a fixed width; widths vary by bank between 22 and 32 characters."""
    s = " ".join(s.upper().split())  # ALL CAPS, collapsed spaces
    return s[: rng.randint(22, 32)].rstrip()


def _render(m: Merchant, d: date, rng: random.Random, home_city: str) -> str:
    mmdd = d.strftime("%m%d")
    name = m.name
    if m.kind == "fee":
        return _truncate(name, rng)
    if m.category == "Income / Deposits":
        if m.kind == "payer":
            ref = rng.choice([f"ST-{_code(rng, 6)}", f"{d:%y%m%d} {_code(rng, 4)}", _digits(rng, 10)])
            return _truncate(f"{name} {ref}", rng)
        return _truncate(rng.choice(ACH_CREDIT).format(m=name), rng)
    if m.category in ("Utilities", "Rent & Lease") or (m.category == "Professional Services" and rng.random() < 0.5):
        return _truncate(rng.choice(ACH_DEBIT).format(m=name) + f" {_digits(rng, rng.randint(0, 8))}", rng)
    if m.category == "Software & Subscriptions":
        suffix = rng.choice(["", "", " APP", ".IO", ".COM", " SUBSCRIPTION", " BILL"])
        tail = rng.choice([f" {_digits(rng, 3)}-{_digits(rng, 3)}-{_digits(rng, 4)}", f" {_code(rng, 8)}", ""])
        return _truncate(f"{rng.choice(SAAS_PROCESSORS)}{name}{suffix}{tail}", rng)
    # Card purchase at a physical or online merchant.
    prefixes = CARD_PREFIXES + (LOCAL_PREFIXES.get(m.category, []) if m.kind == "local" else [])
    prefix = rng.choice(prefixes).format(mmdd=mmdd)
    store = ""
    if m.kind == "chain" and rng.random() < 0.6:
        store = rng.choice([f" #{_digits(rng, 4)}", f" {_digits(rng, 5)}", f" {_digits(rng, 3)}"])
    city = home_city if m.kind == "local" else rng.choice(CITIES)
    # Card descriptors carry a store number and/or a location; never the bare brand alone.
    where = f" {city}" if (not store or rng.random() < 0.75) else ""
    return _truncate(f"{prefix}{name}{store}{where}", rng)


def _amount(category: str, rng: random.Random) -> float:
    if category == "Bank Fees & Charges":
        return -rng.choice(FEE_AMOUNTS)
    mu, sigma = AMOUNT_PARAMS[category]
    value = max(0.5, round(rng.lognormvariate(mu, sigma), 2))
    return value if category == "Income / Deposits" else -value


def _posted_date(category: str, merchant_day: int, rng: random.Random) -> date:
    span = (END - START).days
    d = START + timedelta(days=rng.randint(0, span))
    if category == "Rent & Lease":
        return d.replace(day=rng.randint(1, 3))
    if category == "Software & Subscriptions" or category == "Utilities":
        return d.replace(day=merchant_day)  # billed on the same day each month
    if category == "Bank Fees & Charges":
        return d.replace(day=rng.randint(24, 28))  # end-of-cycle, but not a single giveaway day
    if category == "Meals & Entertainment" and rng.random() < 0.15:
        # Mild weekend lean (Fri-Sun), not a giveaway: about 1.5x a weekday, not 4x.
        target = rng.choice([4, 5, 6])
        return min(END, d + timedelta(days=(target - d.weekday()) % 7))
    return d


def assign_splits(merchants: list[Merchant], seed: int) -> dict[str, str]:
    """Merchant → split, stratified by category: ~70/15/15 of each category's merchants."""
    rng = random.Random(seed)
    by_cat: dict[str, list[str]] = defaultdict(list)
    for m in merchants:
        by_cat[m.category].append(m.name)
    out: dict[str, str] = {}
    for cat in sorted(by_cat):
        names = sorted(by_cat[cat])
        rng.shuffle(names)
        n = len(names)
        n_train = round(n * SPLIT_FRACTIONS[0][1])
        n_val = round(n * SPLIT_FRACTIONS[1][1])
        for i, name in enumerate(names):
            out[name] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return out


def generate(merchants: list[Merchant], n_rows: int = 20_000, seed: int = 42) -> list[Txn]:
    rng = random.Random(seed)
    split_of = assign_splits(merchants, seed)
    by_cat: dict[str, list[Merchant]] = defaultdict(list)
    for m in sorted(merchants, key=lambda m: m.name):
        by_cat[m.category].append(m)
    # Skewed popularity: a few merchants account for most rows (chains more than locals).
    popularity = {
        m.name: rng.lognormvariate(0, 1.0) * (3.0 if m.kind in ("chain", "payer") else 1.0)
        for m in sorted(merchants, key=lambda m: m.name)
    }
    home_city = {m.name: rng.choice(CITIES) for m in sorted(merchants, key=lambda m: m.name)}
    billing_day = {m.name: rng.randint(1, 28) for m in sorted(merchants, key=lambda m: m.name)}

    cats = list(CATEGORY_WEIGHTS)
    weights = [CATEGORY_WEIGHTS[c] for c in cats]
    rows: list[Txn] = []
    for _ in range(n_rows):
        cat = rng.choices(cats, weights)[0]
        pool = by_cat[cat]
        m = rng.choices(pool, [popularity[x.name] for x in pool])[0]
        d = _posted_date(cat, billing_day[m.name], rng)
        desc = _render(m, d, rng, home_city[m.name])
        amount = _amount(cat, rng)
        if amount < 0 and cat != "Bank Fees & Charges" and rng.random() < REFUND_RATE:
            amount = -amount
            desc = _truncate(rng.choice(["RETURN ", "REFUND ", "CREDIT "]) + desc, rng)
        rows.append(Txn(datetime(d.year, d.month, d.day), desc, amount, cat, m.name, split_of[m.name]))
    rows.sort(key=lambda t: (t.posted_at, t.description))
    return rows


def write_csv(rows: list[Txn], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["posted_at", "description", "amount", "category", "split", "merchant"])
        for t in rows:
            w.writerow(
                [
                    t.posted_at.date().isoformat(),
                    t.description,
                    f"{t.amount:.2f}",
                    t.category,
                    t.split,
                    t.merchant,
                ]
            )
