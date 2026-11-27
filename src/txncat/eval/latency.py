"""Per-prediction inference latency for E1–E3 on this machine (SPEC §7 "latency", M5).

One transaction at a time (batch size 1, like a live request), after a short warm-up; p50/p95 over a fixed sample of
test rows. E1/E2 are refit with their already-selected hyperparameters (read from reports/metrics/baseline.json)
because the M2 run did not persist models. E3's latency doesn't depend on the trained weights, only on the
architecture, so the base DistilBERT with a 13-way head is timed (on CPU, and on GPU if one is available).
LLM latencies (E4/E5) were measured during their runs and are read from their metrics files.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import sparse

from txncat.config import REPO_ROOT

SAMPLE = 300
WARMUP = 20


def _time(fn: Callable[[int], object], n: int) -> dict:
    for i in range(min(WARMUP, n)):
        fn(i)
    ms = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(i)
        ms.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(ms)
    return {
        "n": n,
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "mean_ms": float(a.mean()),
    }


def measure(df: pd.DataFrame, log: Callable[[str], None] = print) -> dict:
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.linear_model import LogisticRegression
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier

    from txncat.data.taxonomy import category_names
    from txncat.features.text_features import make_tfidf
    from txncat.models import baseline

    selected = json.loads((REPO_ROOT / "reports/metrics/baseline.json").read_text())["models"]
    train = df[df["split"] == "train"].reset_index(drop=True)
    sample = df[df["split"] == "test"].sample(n=SAMPLE, random_state=0).reset_index(drop=True)
    out: dict = {"sample_rows": SAMPLE, "batch_size": 1, "models": {}}

    tfidf = make_tfidf()
    Xt = sparse.csr_matrix(tfidf.fit_transform(train["desc_clean"]))

    c = selected["E1_logreg_tfidf"]["params"]["C"]
    lr = LogisticRegression(C=c, max_iter=3000, class_weight="balanced", random_state=baseline.SEED).fit(
        Xt, train["category"]
    )
    out["models"]["E1_logreg_tfidf"] = {
        "device": "cpu",
        **_time(lambda i: lr.predict(tfidf.transform(sample["desc_clean"].iloc[[i]])), SAMPLE),
    }
    log(f"E1 p50 {out['models']['E1_logreg_tfidf']['p50_ms']:.2f} ms")

    p2 = selected["E2_xgb_tfidf_sql"]["params"]
    labels = category_names()
    index = {lab: k for k, lab in enumerate(labels)}
    sel = SelectKBest(chi2, k=min(baseline.XGB_TEXT_FEATURES, Xt.shape[1])).fit(Xt, train["category"])
    y = train["category"].map(index).to_numpy()
    xgb = XGBClassifier(
        n_estimators=p2["n_estimators_used"],
        max_depth=p2["max_depth"],
        learning_rate=p2["learning_rate"],
        min_child_weight=p2["min_child_weight"],
        tree_method="hist",
        objective="multi:softprob",
        num_class=len(labels),
        n_jobs=1,
        random_state=baseline.SEED,
    )
    xgb.fit(
        sparse.hstack([sel.transform(Xt), baseline.numeric_matrix(train)], format="csr"),
        y,
        sample_weight=compute_sample_weight("balanced", y),
    )

    def e2(i: int) -> object:
        row = sample.iloc[[i]]
        x = sparse.hstack(
            [sel.transform(tfidf.transform(row["desc_clean"])), baseline.numeric_matrix(row)], format="csr"
        )
        return xgb.predict(x)

    out["models"]["E2_xgb_tfidf_sql"] = {"device": "cpu", **_time(e2, SAMPLE)}
    log(f"E2 p50 {out['models']['E2_xgb_tfidf_sql']['p50_ms']:.2f} ms")

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from txncat.models.finetune import BASE_MODEL, BASE_REVISION
    except ImportError:
        log("E3 skipped: encoder extra not installed")
        return out
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        model = (
            AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL, revision=BASE_REVISION, num_labels=len(labels)
            )
            .to(dev)
            .eval()
        )

        @torch.no_grad()
        def e3(i: int, model=model, dev=dev) -> object:
            enc = tok(sample["desc_clean"].iloc[[i]].tolist(), truncation=True, max_length=32, return_tensors="pt")
            logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits
            return logits.argmax(-1).cpu()

        key = "E3_finetune" + ("" if dev == "cpu" else "_gpu")
        out["models"][key] = {
            "device": dev if dev == "cpu" else str(torch.cuda.get_device_name(0)),
            **_time(e3, SAMPLE),
        }
        log(f"{key} p50 {out['models'][key]['p50_ms']:.2f} ms")
    return out
