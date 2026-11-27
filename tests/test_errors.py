"""M6 error typing rules and the error-analysis pipeline (fake predictions; no database)."""

import numpy as np
import pandas as pd
import pytest

from txncat.data.taxonomy import category_names
from txncat.eval import errors_cli, report
from txncat.eval.errors import TYPE_KEYS, analyze, classify_error

CATS = category_names()


@pytest.mark.parametrize(
    ("desc", "direction", "true", "pred", "expected"),
    [
        ("XFINITY BILL PAY", "debit", "Utilities", "INVALID", "invalid_reply"),
        ("PAYPAL INST XFER VRBO", "debit", "Travel", "Transfer", "invalid_reply"),  # non-category text
        ("HOTWIRE MINNEAPOLIS", "credit", "Travel", "Income / Deposits", "credit_on_expense"),
        ("Incoming Wire Transfer", "credit", "Other", "Income / Deposits", "taxonomy_boundary"),  # Other wins
        ("TRADER JOE'S 6260", "debit", "Other", "Meals & Entertainment", "taxonomy_boundary"),
        ("WITHDRAWAL FROM PG AND E", "debit", "Utilities", "Bank Fees & Charges", "rail_wrapper"),
        ("AT&T EBILL WEB ID: QP02", "debit", "Utilities", "Bank Fees & Charges", "rail_wrapper"),
        ("SQ *THAI BASIL #291", "debit", "Meals & Entertainment", "Other", "hedge_other"),  # pred Other beats rail
        ("MSC CRUISES BOOKING", "debit", "Travel", "Other", "hedge_other"),
        ("NATIONAL CAR RENTAL", "debit", "Travel", "Rent & Lease", "confusion"),
        ("SOMETHING", None, "Travel", "Rent & Lease", "confusion"),  # missing direction is fine
    ],
)
def test_classify_error(desc: str, direction: str | None, true: str, pred: str, expected: str) -> None:
    assert classify_error(desc, direction, true, pred, CATS) == expected


def test_classify_error_rejects_correct_rows() -> None:
    with pytest.raises(ValueError):
        classify_error("X", "debit", "Travel", "Travel", CATS)


def test_analyze_counts_and_fix_rates() -> None:
    rows = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "description": ["SQ *TACO", "WITHDRAWAL FROM PG AND E", "HOTEL", "ATM"],
            "direction": ["debit", "debit", "debit", "debit"],
            "category": ["Meals & Entertainment", "Utilities", "Travel", "Other"],
        },
        index=[10, 11, 12, 13],  # non-default index must not matter
    )
    preds = {
        "E5": np.array(["Other", "Bank Fees & Charges", "Travel", "Bank Fees & Charges"]),
        "E2": np.array(["Meals & Entertainment", "Bank Fees & Charges", "Travel", "Other"]),
    }
    res = analyze(rows, preds, "E5", CATS)
    assert res["n_errors"] == 3
    assert res["counts"] == {**dict.fromkeys(TYPE_KEYS, 0), "hedge_other": 1, "rail_wrapper": 1, "taxonomy_boundary": 1}
    assert res["fixed_by"]["hedge_other"]["E2"] == 1.0
    assert res["fixed_by"]["rail_wrapper"]["E2"] == 0.0
    assert np.isnan(res["fixed_by"]["confusion"]["E2"])
    assert len(res["sample"]) == 3 and set(res["sample"]["E2_right"]) == {True, False}


def test_build_and_overuse_on_fake_prediction_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    n = 60
    cats = np.array(CATS[:5])
    y = cats[rng.integers(0, 5, n)]
    desc = np.array([("SQ *SHOP " if i % 3 == 0 else "SHOP ") + str(i) for i in range(n)])
    texts = []
    for split, offset in (("test", 0), ("external_test", 1000)):
        ids = np.arange(n) + offset
        texts.append(pd.DataFrame({"id": ids, "split": split, "description": desc, "direction": "debit"}))
        for _label, files in report.MODELS.values():
            for f in files:
                pred = np.where(rng.random(n) < 0.7, y, "Other")
                part = pd.DataFrame({"id": ids, "split": split, "category": y, "pred": pred})
                path = tmp_path / f
                old = pd.read_csv(path) if path.exists() else None
                pd.concat([old, part] if old is not None else [part]).to_csv(path, index=False)
    t = pd.concat(texts)
    res, sample = errors_cli.build(t, tmp_path)
    assert res["n_rows"] == n and res["n_errors"] == sum(res["counts"].values())
    assert "E3" in res["fixed_by"]["hedge_other"]  # mean over seeds added
    assert len(sample) == min(30, res["n_errors"])
    ov = errors_cli.other_overuse(t, tmp_path)
    assert ov["external_test"]["E5"]["pos_prefix"]["rows"] + ov["external_test"]["E5"]["no_prefix"]["rows"] == int(
        np.sum(y != "Other")
    )
