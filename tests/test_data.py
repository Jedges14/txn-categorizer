"""M1 data tests: taxonomy, merchant list, synthetic generator, merchant-grouped splits, real-test template."""

import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pytest

from txncat.config import REPO_ROOT
from txncat.data.generate_synthetic import (
    AMOUNT_PARAMS,
    CATEGORY_WEIGHTS,
    END,
    START,
    Txn,
    generate,
    write_csv,
)
from txncat.data.load_to_postgres import RealTestError, read_real_test_csv
from txncat.data.merchants import build_merchants, read_merchants_csv
from txncat.data.report import class_distribution_chart, counts_from_csv, markdown_table
from txncat.data.taxonomy import category_names, load_taxonomy

SEED = 42


@pytest.fixture(scope="module")
def merchants():
    return build_merchants(seed=SEED)


@pytest.fixture(scope="module")
def rows(merchants) -> list[Txn]:
    return generate(merchants, n_rows=20_000, seed=SEED)  # exactly what `txncat data build` makes


# --- taxonomy ---------------------------------------------------------------------------------------------
def test_taxonomy_has_13_documented_categories() -> None:
    cats = load_taxonomy()
    assert len(cats) == 13
    assert len({c.name for c in cats}) == 13
    assert "Other" in {c.name for c in cats}
    for c in cats:
        assert c.definition.strip()
        assert len(c.examples) == 3


def test_generator_covers_exactly_the_taxonomy() -> None:
    names = set(category_names())
    assert set(CATEGORY_WEIGHTS) == names
    assert set(AMOUNT_PARAMS) | {"Bank Fees & Charges"} == names


# --- merchants --------------------------------------------------------------------------------------------
def test_merchant_names_unique_and_plentiful(merchants) -> None:
    names = [m.name for m in merchants]
    assert len(names) == len(set(names)), "a merchant name appears in two categories"
    per_cat = Counter(m.category for m in merchants)
    assert set(per_cat) == set(category_names())
    assert min(per_cat.values()) >= 250, per_cat


def test_committed_merchants_csv_is_up_to_date(merchants) -> None:
    committed = read_merchants_csv(REPO_ROOT / "data" / "merchants.csv")
    assert sorted(committed, key=lambda m: m.name) == sorted(merchants, key=lambda m: m.name), (
        "data/merchants.csv is stale: run `txncat data build`"
    )


# --- generator --------------------------------------------------------------------------------------------
def test_generation_is_deterministic(merchants) -> None:
    a = generate(merchants, n_rows=500, seed=7)
    b = generate(merchants, n_rows=500, seed=7)
    c = generate(merchants, n_rows=500, seed=8)
    assert a == b
    assert a != c


def test_no_merchant_appears_in_two_splits(rows) -> None:
    splits_of: dict[str, set[str]] = defaultdict(set)
    for t in rows:
        splits_of[t.merchant].add(t.split)
    leaking = {m: s for m, s in splits_of.items() if len(s) > 1}
    assert not leaking, f"{len(leaking)} merchants in more than one split, e.g. {list(leaking)[:3]}"


def test_every_split_has_every_category(rows) -> None:
    cats = set(category_names())
    for split in ("train", "val", "test"):
        assert {t.category for t in rows if t.split == split} == cats, split


def test_split_proportions_are_roughly_70_15_15(rows) -> None:
    share = {s: n / len(rows) for s, n in Counter(t.split for t in rows).items()}
    assert 0.62 <= share["train"] <= 0.78
    assert 0.10 <= share["val"] <= 0.20
    assert 0.10 <= share["test"] <= 0.20


def test_classes_are_imbalanced_on_purpose(rows) -> None:
    counts = Counter(t.category for t in rows)
    assert max(counts.values()) / min(counts.values()) >= 3


def test_descriptions_look_like_bank_strings(rows) -> None:
    for t in rows:
        d = t.description
        assert 3 <= len(d) <= 32, d
        assert d == d.upper(), d
        assert "  " not in d and d == d.strip(), repr(d)


def test_toast_prefix_only_on_restaurants(rows) -> None:
    assert {t.category for t in rows if re.match(r"^(\w+ )?TST\* ", t.description)} <= {"Meals & Entertainment"}


def test_amount_signs_and_cents(rows) -> None:
    for t in rows:
        assert round(t.amount, 2) == t.amount
        if t.category == "Income / Deposits":
            assert t.amount > 0, t
        elif t.amount > 0:  # only refunds/returns of expenses are positive
            assert t.description.startswith(("RETURN ", "REFUND ", "CREDIT ")), t


def test_dates_in_range(rows) -> None:
    lo, hi = datetime(START.year, START.month, START.day), datetime(END.year, END.month, END.day)
    assert all(lo <= t.posted_at <= hi for t in rows)


# --- report -----------------------------------------------------------------------------------------------
def test_report_table_and_chart(rows, tmp_path: Path) -> None:
    csv_path = tmp_path / "synthetic.csv"
    write_csv(rows, csv_path)
    counts, merchant_counts = counts_from_csv(csv_path)
    assert sum(counts.values()) == len(rows)
    table = markdown_table(counts, merchant_counts)
    assert "| **All** |" in table and "Bank Fees & Charges" in table
    png = class_distribution_chart(counts, tmp_path / "chart.png")
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --- real test template -----------------------------------------------------------------------------------
def test_sample_real_test_file_matches_template() -> None:
    rows = read_real_test_csv(REPO_ROOT / "data" / "sample" / "real_test_example.csv")
    assert len(rows) == 20
    assert {r["split"] for r in rows} == {"real_test"}
    assert {r["source"] for r in rows} == {"real"}


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("date,desc,amount,category\n", "header must be exactly"),
        (
            "posted_at,description,amount,category\n2025-01-02,X,-1.00,Food\n",
            "line 2: unknown category",
        ),
        ("posted_at,description,amount,category\n01/02/2025,X,-1.00,Other\n", "line 2: posted_at"),
        ("posted_at,description,amount,category\n2025-01-02,X,$1.00,Other\n", "line 2: amount"),
        (
            "posted_at,description,amount,category\n2025-01-02, ,-1.00,Other\n",
            "line 2: description",
        ),
    ],
)
def test_real_test_validation_reports_line_numbers(tmp_path: Path, content: str, message: str) -> None:
    p = tmp_path / "real_test.csv"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(RealTestError, match=message):
        read_real_test_csv(p)
