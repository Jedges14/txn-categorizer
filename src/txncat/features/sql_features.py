"""Read the SQL feature view (sql/features.sql) into pandas."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine, text

# Numeric/boolean features passed to tree models. NaN where unknown (e.g. no amount/date on external rows).
NUMERIC_FEATURES = [
    "amount_abs_log",
    "is_outflow",
    "amount_bucket",
    "day_of_week",
    "is_weekend",
    "day_of_month",
    "merchant_freq_log",
    "has_processor_prefix",
    "is_refund",
    "desc_len",
    "digit_ratio",
]


def load_features(engine: Engine) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM txn_features ORDER BY id"), conn)
    return add_derived(df)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amount_abs_log"] = np_log1p(df["amount_abs"])
    df["merchant_freq_log"] = np_log1p(df["merchant_freq"])
    for col in ("is_outflow", "is_weekend", "has_processor_prefix", "is_refund"):
        df[col] = df[col].astype("float64")  # bool/None → 1.0/0.0/NaN
    for col in ("amount_bucket", "day_of_week", "day_of_month", "desc_len", "digit_ratio"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def np_log1p(s: pd.Series) -> pd.Series:
    import numpy as np

    return np.log1p(pd.to_numeric(s, errors="coerce").astype("float64"))
