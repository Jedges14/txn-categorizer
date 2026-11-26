"""Metrics, keyword rules, feature matrix and a fast end-to-end baseline smoke test (no database)."""

import numpy as np
import pandas as pd
import pytest

from txncat.data.generate_synthetic import generate
from txncat.data.merchants import build_merchants
from txncat.eval.metrics import confusion, score
from txncat.features.sql_features import NUMERIC_FEATURES, add_derived
from txncat.models import baseline


def test_score_macro_f1_averages_over_present_classes_only() -> None:
    s = score(["A", "A", "B", "B"], ["A", "A", "B", "C"])  # C is predicted but never true
    assert s.labels == ["A", "B"]
    assert s.accuracy == pytest.approx(0.75)
    assert s.per_class_f1 == pytest.approx({"A": 1.0, "B": 2 / 3})
    assert s.macro_f1 == pytest.approx((1.0 + 2 / 3) / 2)


def test_score_rejects_empty() -> None:
    with pytest.raises(ValueError):
        score([], [])


def test_confusion_counts_predictions_outside_true_labels() -> None:
    m = confusion(["A", "B"], ["C", "B"], labels=["A", "B", "C"])
    assert m.tolist() == [[0, 0, 1], [0, 1, 0], [0, 0, 0]]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("CHASE MONTHLY SERVICE FEE", "Bank Fees & Charges"),
        ("STRIPE TRANSFER ST-X9K2L", "Income / Deposits"),
        ("TAMPA SELF STORAGE AUTOPAY", "Rent & Lease"),
        ("FACEBK ADS", "Advertising & Marketing"),
        ("FEDEX OFFICE", "Shipping & Postage"),  # rules are ordered: Shipping before Office Supplies
        ("SOMETHING UNKNOWN", "Other"),  # fallback
    ],
)
def test_keyword_rules(text: str, expected: str) -> None:
    assert baseline.predict_rules(pd.Series([text]), fallback="Other")[0] == expected


def test_numeric_matrix_keeps_zeros_and_leaves_nan_missing() -> None:
    df = pd.DataFrame({c: [0.0, np.nan] for c in NUMERIC_FEATURES})
    m = baseline.numeric_matrix(df)
    assert m.shape == (2, len(NUMERIC_FEATURES))
    assert m[0].nnz == len(NUMERIC_FEATURES) and m[0].toarray().min() == 1.0  # real zeros stored as 1.0
    assert m[1].nnz == 0  # unknown values stay missing for XGBoost


def _frame_from_generator(n: int) -> pd.DataFrame:
    rows = generate(build_merchants(), n_rows=n, seed=5)
    df = pd.DataFrame(
        {
            "id": range(len(rows)),
            "split": [t.split for t in rows],
            "category": [t.category for t in rows],
            "description": [t.description for t in rows],
            "desc_clean": [t.description for t in rows],  # good enough for a smoke test
            "amount_abs": [abs(t.amount) for t in rows],
            "is_outflow": [t.amount < 0 for t in rows],
            "amount_bucket": [min(10, int(abs(t.amount) // 50)) for t in rows],
            "day_of_week": [t.posted_at.isoweekday() for t in rows],
            "is_weekend": [t.posted_at.isoweekday() >= 6 for t in rows],
            "day_of_month": [t.posted_at.day for t in rows],
            "merchant_freq": [0] * len(rows),
            "has_processor_prefix": ["*" in t.description for t in rows],
            "is_refund": [t.description.startswith("REFUND") for t in rows],
            "desc_len": [len(t.description) for t in rows],
            "digit_ratio": [sum(ch.isdigit() for ch in t.description) / len(t.description) for t in rows],
        }
    )
    return add_derived(df)


def test_run_baselines_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(baseline, "LOGREG_C", [1.0])
    monkeypatch.setattr(baseline, "XGB_GRID", {"max_depth": [3], "learning_rate": [0.3], "min_child_weight": [1]})
    df = _frame_from_generator(1500)
    results = baseline.run_baselines(df, eval_splits=("val", "test"))
    names = [r.name for r in results]
    assert names == ["E0_majority", "E0_rules", "E1_logreg_tfidf", "E2_xgb_tfidf_sql"]
    n_eval = int(df["split"].isin(["val", "test"]).sum())
    for r in results:
        assert len(r.predictions) == n_eval
        assert set(r.predictions["split"]) <= {"val", "test"}
    learned = {r.name: r.val_macro_f1 for r in results}
    assert learned["E1_logreg_tfidf"] > learned["E0_majority"]  # learning beats a constant
    assert learned["E2_xgb_tfidf_sql"] > learned["E0_majority"]


def test_run_baselines_never_sees_test_labels_during_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scrambling test labels must not change the selected hyperparameters (they're chosen on val only)."""
    monkeypatch.setattr(baseline, "LOGREG_C", [0.3, 3.0])
    monkeypatch.setattr(baseline, "XGB_GRID", {"max_depth": [3], "learning_rate": [0.3], "min_child_weight": [1]})
    df = _frame_from_generator(1200)
    scrambled = df.copy()
    test_mask = scrambled["split"] == "test"
    scrambled.loc[test_mask, "category"] = (
        scrambled.loc[test_mask, "category"].sample(frac=1, random_state=0).to_numpy()
    )
    a = {r.name: r.params for r in baseline.run_baselines(df, eval_splits=("test",))}
    b = {r.name: r.params for r in baseline.run_baselines(scrambled, eval_splits=("test",))}
    assert a == b
