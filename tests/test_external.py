"""External test set: mapping, exclusions, direction parsing, sampling, checksum. No network: local fixtures only."""

import csv
from collections import Counter
from pathlib import Path

import pytest

from txncat.data.external import (
    CATEGORY_MAP,
    EXCLUDED,
    ExternalDataError,
    build_external_test,
    download,
    overlap_with_merchants,
    sha256_of,
)
from txncat.data.taxonomy import category_names


@pytest.fixture
def fixture_csv(tmp_path: Path) -> Path:
    """A miniature file in the external dataset's format (description, category)."""
    rows = []
    for cat in list(CATEGORY_MAP) + list(EXCLUDED):
        for i in range(12):
            sign = "[credit]" if cat == "Income" else "[debit]"
            rows.append((f"{sign} {cat.upper()} MERCHANT {i:02d}", cat))
    rows.append(("[debit] RESTAURANTS MERCHANT 00", "Restaurants"))  # exact duplicate string
    path = tmp_path / "ext.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["description", "category"])
        w.writerows(rows)
    return path


def test_mapping_targets_are_real_categories_and_sources_dont_overlap() -> None:
    assert set(CATEGORY_MAP.values()) <= set(category_names())
    assert not set(CATEGORY_MAP) & set(EXCLUDED)


def test_excluded_categories_never_appear(fixture_csv: Path) -> None:
    rows = build_external_test(fixture_csv, per_category=8, seed=1)
    descriptions = " ".join(r["description"] for r in rows)
    for cat in EXCLUDED:
        assert f"{cat.upper()} MERCHANT" not in descriptions


def test_rows_are_labeled_split_and_stripped(fixture_csv: Path) -> None:
    rows = build_external_test(fixture_csv, per_category=8, seed=1)
    assert {r["split"] for r in rows} == {"external_test"}
    assert all(r["amount"] is None and r["posted_at"] is None for r in rows)  # never invented
    assert all(not r["description"].startswith("[") for r in rows)
    income = [r for r in rows if r["category"] == "Income / Deposits"]
    assert income and all(r["direction"] == "credit" for r in income)
    assert all(r["direction"] == "debit" for r in rows if r["category"] != "Income / Deposits")


def test_quota_per_target_spread_over_sources(fixture_csv: Path) -> None:
    rows = build_external_test(fixture_csv, per_category=8, seed=1)
    counts = Counter(r["category"] for r in rows)
    assert counts == {target: 8 for target in set(CATEGORY_MAP.values())}
    other_sources = Counter(r["description"].split(" MERCHANT")[0] for r in rows if r["category"] == "Other")
    assert set(other_sources.values()) == {2}  # 8 rows over 4 source categories


def test_sampling_is_deterministic_and_deduplicated(fixture_csv: Path) -> None:
    a = build_external_test(fixture_csv, per_category=12, seed=3)
    assert a == build_external_test(fixture_csv, per_category=12, seed=3)
    assert a != build_external_test(fixture_csv, per_category=8, seed=4)
    meals = [r["description"] for r in a if r["category"] == "Meals & Entertainment"]
    assert len(meals) == len(set(meals)) == 12


def test_download_rejects_checksum_mismatch(tmp_path: Path) -> None:
    src = tmp_path / "src.csv"
    src.write_text("description,category\n", encoding="utf-8")
    dest = tmp_path / "out" / "file.csv"
    with pytest.raises(ExternalDataError, match="checksum mismatch"):
        download(dest, url=src.resolve().as_uri(), expected_sha256="0" * 64)
    assert not dest.exists()
    ok = download(dest, url=src.resolve().as_uri(), expected_sha256=sha256_of(src))
    assert ok.read_text(encoding="utf-8") == "description,category\n"


def test_overlap_finds_whole_word_merchant_mentions() -> None:
    rows = [{"description": d} for d in ("Chipotle 1234 Austin", "CHIPOTLEX 9", "bp gas", "Unknown Cafe")]
    found = overlap_with_merchants(rows, {"CHIPOTLE", "BP"})
    assert found == ["CHIPOTLE", None, None, None]  # case-insensitive, whole words, short names skipped
