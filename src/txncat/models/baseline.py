"""Baselines E0–E2 (SPEC §7).

E0  majority class, and hand-written keyword rules (from the taxonomy definitions, not tuned on any data)
E1  logistic regression on TF-IDF(desc_clean)
E2  XGBoost on the top-5k chi2 TF-IDF columns (selected on train) + SQL features

Hyperparameters are chosen on the VAL split only. Test splits are scored once, after selection.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from txncat.data.taxonomy import category_names
from txncat.eval.metrics import score
from txncat.features.sql_features import NUMERIC_FEATURES
from txncat.features.text_features import make_tfidf

SEED = 42

# Ordered keyword rules: first match wins. Written from data/taxonomy.yaml definitions and general knowledge of
# statements; never adjusted after looking at val/test/external results.
KEYWORD_RULES: list[tuple[str, str]] = [
    ("Bank Fees & Charges", r"\bFEE\b|\bNSF\b|OVERDRAFT"),
    ("Income / Deposits", r"\bDEPOSIT\b|PAYOUT|ACH CREDIT|ZELLE FROM|WIRE IN|DIR DEP|PAYROLL|TRANSFER ST"),
    ("Rent & Lease", r"\bRENT\b|LEASE|LEASING|STORAGE|COWORK|PROPERT|REALTY|WEWORK|REGUS"),
    ("Advertising & Marketing", r"\bADS\b|ADVERTIS|MARKETING|PROMO|PRINTING|\bSIGN\b|BILLBOARD|VISTAPRINT"),
    (
        "Utilities",
        r"ELECTRIC|\bPOWER\b|\bWATER\b|\bGAS & ELECTRIC\b|UTILIT|COMCAST|XFINITY|VERIZON|AT&T"
        r"|INTERNET|FIBER|TELEPHONE|WASTE",
    ),
    (
        "Professional Services",
        r"\bCPA\b|\bLAW\b|\bLLP\b|LEGAL|CONSULT|BOOKKEEP|ACCOUNTING|\bTAX\b|STAFFING|PAYROLL SVC|GUSTO|UPWORK",
    ),
    ("Shipping & Postage", r"\bUSPS\b|\bUPS\b|FEDEX|\bDHL\b|SHIP|POSTAL|POSTAGE|COURIER|FREIGHT|\bMAIL\b"),
    (
        "Travel",
        r"AIRLINE|\bAIR\b|HOTEL|\bINN\b|SUITES|LODGE|MARRIOTT|HILTON|AIRBNB|UBER|LYFT|AMTRAK"
        r"|RENT-A-CAR|CAR RENTAL|PARKING|TRAVEL",
    ),
    ("Fuel & Auto", r"\bGAS\b|FUEL|SHELL|CHEVRON|EXXON|\bBP\b|AUTO|TIRE|LUBE|CAR WASH|COLLISION|EZPASS|FASTRAK"),
    (
        "Software & Subscriptions",
        r"SOFTWARE|\bAPP\b|\bIO\b|ADOBE|GOOGLE|MICROSOFT|DROPBOX|SLACK|ZOOM|GITHUB|SUBSCRIPTION|\bAWS\b|PADDLE|SAAS",
    ),
    ("Office Supplies", r"OFFICE|STAPLES|STATIONERY|\bPAPER\b|HARDWARE|COMPUTER|FURNITURE|BEST BUY"),
    (
        "Meals & Entertainment",
        r"CAFE|COFFEE|PIZZA|GRILL|KITCHEN|BISTRO|DINER|TAVERN|BAKERY|RESTAURANT|STARBUCKS|DOORDASH"
        r"|UBER EATS|TACO|SUSHI|BURGER|DELI|\bBBQ\b|TICKET",
    ),
]
_COMPILED_RULES = [(cat, re.compile(p)) for cat, p in KEYWORD_RULES]

LOGREG_C = [0.3, 1.0, 3.0, 10.0]
XGB_TEXT_FEATURES = 5000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


XGB_GRID: dict[str, list[float]] = {"max_depth": [4, 6, 8], "learning_rate": [0.1, 0.3], "min_child_weight": [1, 3]}


@dataclass
class ModelResult:
    name: str
    params: dict
    val_macro_f1: float
    predictions: pd.DataFrame  # id, split, category, pred
    fit_seconds: float
    tuning: list[dict] = field(default_factory=list)


def predict_rules(texts: pd.Series, fallback: str) -> np.ndarray:
    out = []
    for t in texts.fillna(""):
        out.append(next((cat for cat, rx in _COMPILED_RULES if rx.search(t)), fallback))
    return np.asarray(out)


def numeric_matrix(df: pd.DataFrame) -> sparse.csr_matrix:
    """SQL features as sparse columns. Values are shifted by +1 so that a real 0 is stored (XGBoost treats an
    unstored sparse entry as MISSING); NaN (unknown, e.g. no amount on external rows) stays unstored."""
    vals = df[NUMERIC_FEATURES].to_numpy(dtype="float64") + 1.0
    rows, cols = np.nonzero(~np.isnan(vals))
    return sparse.csr_matrix((vals[rows, cols], (rows, cols)), shape=vals.shape)


def _frame(df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": df["id"].to_numpy(),
            "split": df["split"].to_numpy(),
            "category": df["category"].to_numpy(),
            "pred": pred,
        }
    )


def run_baselines(
    df: pd.DataFrame, eval_splits: tuple[str, ...] = ("val", "test", "external_test")
) -> list[ModelResult]:
    """Train E0–E2 on split == 'train', select on 'val', predict every split in `eval_splits`."""
    labels = category_names()
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    evals = df[df["split"].isin(eval_splits)].reset_index(drop=True)
    if train.empty or val.empty:
        raise ValueError("need train and val rows; run `txncat data load` first")
    results: list[ModelResult] = []

    # --- E0: majority class and keyword rules --------------------------------------------------------------
    majority = Counter(train["category"]).most_common(1)[0][0]
    t0 = time.perf_counter()
    results.append(
        ModelResult(
            "E0_majority",
            {"class": majority},
            score(val["category"], [majority] * len(val)).macro_f1,
            _frame(evals, np.full(len(evals), majority)),
            time.perf_counter() - t0,
        )
    )
    t0 = time.perf_counter()
    results.append(
        ModelResult(
            "E0_rules",
            {"rules": len(KEYWORD_RULES), "fallback": majority},
            score(val["category"], predict_rules(val["desc_clean"], majority)).macro_f1,
            _frame(evals, predict_rules(evals["desc_clean"], majority)),
            time.perf_counter() - t0,
        )
    )

    # --- shared TF-IDF, fit on train text only ---------------------------------------------------------------
    tfidf = make_tfidf()
    Xt_train = sparse.csr_matrix(tfidf.fit_transform(train["desc_clean"]))
    Xt_val = sparse.csr_matrix(tfidf.transform(val["desc_clean"]))
    Xt_eval = sparse.csr_matrix(tfidf.transform(evals["desc_clean"]))

    # --- E1: logistic regression, C chosen on val -----------------------------------------------------------
    t0 = time.perf_counter()
    tuning, best = [], None
    for c in LOGREG_C:
        m = LogisticRegression(C=c, max_iter=3000, class_weight="balanced", random_state=SEED)
        m.fit(Xt_train, train["category"])
        f1 = score(val["category"], m.predict(Xt_val)).macro_f1
        tuning.append({"C": c, "val_macro_f1": f1})
        log(f"E1 C={c} val macro-F1={f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, c, m)
    assert best is not None
    results.append(
        ModelResult(
            "E1_logreg_tfidf",
            {"C": best[1], "class_weight": "balanced"},
            best[0],
            _frame(evals, best[2].predict(Xt_eval)),
            time.perf_counter() - t0,
            tuning,
        )
    )

    # --- E2: XGBoost on TF-IDF + SQL features, small grid on val -------------------------------------------
    # Trees over ~50k sparse TF-IDF columns are very slow (a first run took >25 min without finishing), so E2 keeps
    # the XGB_TEXT_FEATURES most class-informative text columns by chi-squared, selected on TRAIN rows only.
    selector = SelectKBest(chi2, k=min(XGB_TEXT_FEATURES, Xt_train.shape[1])).fit(Xt_train, train["category"])
    X_train = sparse.hstack([selector.transform(Xt_train), numeric_matrix(train)], format="csr")
    X_val = sparse.hstack([selector.transform(Xt_val), numeric_matrix(val)], format="csr")
    X_eval = sparse.hstack([selector.transform(Xt_eval), numeric_matrix(evals)], format="csr")
    index = {lab: i for i, lab in enumerate(labels)}
    y_train = train["category"].map(index).to_numpy()
    y_val = val["category"].map(index).to_numpy()
    weights = compute_sample_weight("balanced", y_train)

    t0 = time.perf_counter()
    tuning, best_x = [], None
    keys = list(XGB_GRID)
    combos = list(product(*(XGB_GRID[k] for k in keys)))
    for i, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo, strict=True))
        c0 = time.perf_counter()
        m = XGBClassifier(
            n_estimators=400,
            early_stopping_rounds=20,
            tree_method="hist",
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=len(labels),
            n_jobs=-1,
            random_state=SEED,
            **params,
        )
        m.fit(X_train, y_train, sample_weight=weights, eval_set=[(X_val, y_val)], verbose=False)
        pred_val = np.asarray(labels)[m.predict(X_val)]
        f1 = score(val["category"], pred_val).macro_f1
        tuning.append({**params, "best_iteration": int(m.best_iteration), "val_macro_f1": f1})
        log(
            f"E2 {i}/{len(combos)} {params} trees={int(m.best_iteration) + 1} "
            f"val macro-F1={f1:.4f} ({time.perf_counter() - c0:.0f}s)"
        )
        if best_x is None or f1 > best_x[0]:
            best_x = (f1, {**params, "n_estimators_used": int(m.best_iteration) + 1}, m)
    assert best_x is not None
    pred_eval = np.asarray(labels)[best_x[2].predict(X_eval)]
    results.append(
        ModelResult(
            "E2_xgb_tfidf_sql", best_x[1], best_x[0], _frame(evals, pred_eval), time.perf_counter() - t0, tuning
        )
    )
    return results
