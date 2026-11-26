"""Post-hoc DIAGNOSTIC (not model selection): why does E2 collapse on the external test set?

E2 (XGBoost, TF-IDF + SQL features) beat E1 on the synthetic test split but fell below the keyword rules on the
external set, predicting "Professional Services" for about half of it. This refits E2 with its ALREADY-SELECTED
hyperparameters (read from reports/metrics/baseline.json; no new tuning) while removing one feature group at a time,
and reports test and external scores. The headline E2 result is not replaced; this only explains it.

Variants:
  no_format       drop desc_len, digit_ratio, has_processor_prefix (string-format features)
  no_amount_date  drop amount and calendar features (always present in training, always missing on external rows)
  text_only       TF-IDF columns only

Usage: uv run python scripts/diagnose_e2_external.py
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from txncat.config import REPO_ROOT
from txncat.data.taxonomy import category_names
from txncat.db import make_engine
from txncat.eval.metrics import score
from txncat.features.sql_features import NUMERIC_FEATURES, load_features
from txncat.features.text_features import make_tfidf
from txncat.models import baseline

AMOUNT_DATE = ["amount_abs_log", "amount_bucket", "day_of_week", "is_weekend", "day_of_month"]
FORMAT = ["desc_len", "digit_ratio", "has_processor_prefix"]
VARIANTS: dict[str, list[str]] = {
    "no_format": [f for f in NUMERIC_FEATURES if f not in FORMAT],
    "no_amount_date": [f for f in NUMERIC_FEATURES if f not in AMOUNT_DATE],
    "text_only": [],
}
OUT = REPO_ROOT / "reports" / "metrics" / "e2_external_diagnostic.json"


def main() -> None:
    selected = json.loads((REPO_ROOT / "reports/metrics/baseline.json").read_text())["models"]["E2_xgb_tfidf_sql"]
    params = {k: v for k, v in selected["params"].items() if k in baseline.XGB_GRID}
    n_trees = selected["params"]["n_estimators_used"]

    df = load_features(make_engine())
    labels = category_names()
    index = {lab: i for i, lab in enumerate(labels)}
    train = df[df["split"] == "train"].reset_index(drop=True)
    ev = df[df["split"].isin(["test", "external_test"])].reset_index(drop=True)
    is_ext = (ev["split"] == "external_test").to_numpy()

    tfidf = make_tfidf()
    Xt_train = sparse.csr_matrix(tfidf.fit_transform(train["desc_clean"]))
    Xt_ev = sparse.csr_matrix(tfidf.transform(ev["desc_clean"]))
    sel = SelectKBest(chi2, k=min(baseline.XGB_TEXT_FEATURES, Xt_train.shape[1])).fit(Xt_train, train["category"])
    T_train, T_ev = sel.transform(Xt_train), sel.transform(Xt_ev)
    y = train["category"].map(index).to_numpy()
    weights = compute_sample_weight("balanced", y)

    report: dict = {"params": params, "n_estimators": n_trees, "variants": {}}
    original = baseline.NUMERIC_FEATURES
    for name, features in VARIANTS.items():
        try:
            baseline.NUMERIC_FEATURES = features  # numeric_matrix reads this module attribute
            parts_train = [T_train] + ([baseline.numeric_matrix(train)] if features else [])
            parts_ev = [T_ev] + ([baseline.numeric_matrix(ev)] if features else [])
        finally:
            baseline.NUMERIC_FEATURES = original
        m = XGBClassifier(
            n_estimators=n_trees,
            tree_method="hist",
            objective="multi:softprob",
            num_class=len(labels),
            n_jobs=-1,
            random_state=baseline.SEED,
            **params,
        )
        m.fit(sparse.hstack(parts_train, format="csr"), y, sample_weight=weights)
        pred = np.asarray(labels)[m.predict(sparse.hstack(parts_ev, format="csr"))]
        test_s = score(ev.loc[~is_ext, "category"], pred[~is_ext])
        ext_s = score(ev.loc[is_ext, "category"], pred[is_ext])
        top = Counter(pred[is_ext].tolist()).most_common(3)
        report["variants"][name] = {
            "numeric_features": features,
            "test": test_s.as_dict(),
            "external_test": ext_s.as_dict(),
            "external_top_predictions": dict(top),
        }
        print(f"{name:<15} test macro-F1 {test_s.macro_f1:.3f} | external macro-F1 {ext_s.macro_f1:.3f} | top {top}")
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
