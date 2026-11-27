"""M5 statistics and the `txncat report` pipeline, on small hand-checkable inputs (no database, no models)."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest

from txncat.data.taxonomy import category_names
from txncat.eval import report, significance
from txncat.eval.significance import bootstrap_diff, bootstrap_macro_f1, holm, macro_f1, mcnemar


def test_mcnemar_counts_discordant_pairs_and_uses_exact_binomial() -> None:
    y = ["A"] * 10
    a = ["A"] * 8 + ["B"] * 2  # right on rows 0-7
    b = ["A"] * 3 + ["B"] * 7  # right on rows 0-2
    m = mcnemar(y, a, b)
    assert (m.a_right_b_wrong, m.a_wrong_b_right) == (5, 0)
    assert m.p_value == pytest.approx(binomtest(5, 5, 0.5).pvalue)  # 2 * 0.5**5 = 0.0625
    assert m.p_value == pytest.approx(0.0625)


def test_mcnemar_identical_models_p_is_one() -> None:
    assert mcnemar(["A", "B"], ["A", "A"], ["A", "A"]).p_value == 1.0


def test_mcnemar_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError):
        mcnemar(["A", "B"], ["A"], ["A", "B"])


def test_holm_matches_textbook_example() -> None:
    # sorted p = .01, .02, .03 → ×3, ×2, ×1 = .03, .04, .03 → monotone .03, .04, .04
    assert holm([0.03, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    assert holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])  # capped at 1


def _toy(n: int = 400, seed: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.array(category_names()[:4])  # real names, so the confusion-matrix figure has them
    y = labels[rng.integers(0, 4, n)]
    good = np.where(rng.random(n) < 0.85, y, labels[rng.integers(0, 4, n)])
    bad = np.where(rng.random(n) < 0.5, y, "INVALID")  # off-label predictions must count as errors
    return y, good, bad


def test_fast_bootstrap_f1_equals_sklearn_on_each_resample() -> None:
    y, good, bad = _toy()
    yc, (gc, bc), k = significance._encode(y, good, bad)
    idx = significance._indices(len(y), 5, seed=3)
    for pred, code in ((good, gc), (bad, bc)):
        fast = significance._boot_f1(yc, code, k, idx)
        # sklearn with the classes of the FULL y fixed (a class absent from a resample scores 0)
        from sklearn.metrics import f1_score

        ref = [f1_score(y[i], pred[i], labels=sorted(set(y)), average="macro", zero_division=0) for i in idx]
        assert fast == pytest.approx(ref)


def test_bootstrap_ci_contains_point_and_is_seeded() -> None:
    y, good, _ = _toy()
    f1, lo, hi = bootstrap_macro_f1(y, good, n_boot=300)
    assert f1 == pytest.approx(macro_f1(y, good))
    assert lo < f1 < hi
    assert bootstrap_macro_f1(y, good, n_boot=300) == (f1, lo, hi)  # deterministic


def test_paired_bootstrap_diff_sign() -> None:
    y, good, bad = _toy()
    d, lo, hi = bootstrap_diff(y, good, bad, n_boot=300)
    assert d == pytest.approx(macro_f1(y, good) - macro_f1(y, bad))
    assert 0 < lo < d < hi


def _write_preds(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    y, good, bad = _toy(200)
    ids = np.arange(len(y))
    rng = np.random.default_rng(7)
    for model, (_, files) in report.MODELS.items():
        for f in files:
            pred = good if model in ("E2", "E3") else bad
            if model == "E3":
                pred = np.where(rng.random(len(y)) < 0.9, pred, y[0])
            rows = []
            for split in ("val", "test", "external_test"):
                if split == "val" and model in ("E4", "E5"):
                    continue
                rows.append(
                    pd.DataFrame(
                        {"id": ids + (0 if split == "test" else 1000), "split": split, "category": y, "pred": pred}
                    )
                )
            pd.concat(rows).sample(frac=1, random_state=0).to_csv(tmp_path / f, index=False)  # unsorted on purpose
    monkeypatch.setattr(report, "PRED_DIR", tmp_path)


def test_load_split_aligns_rows_by_id(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _write_preds(tmp_path, monkeypatch)
    p = report.load_split("test", tmp_path)
    assert list(p.ids) == sorted(p.ids)
    assert set(p.cols) == set(report.MODELS)
    assert len(p.cols["E3"]) == 3


def test_load_split_refuses_mismatched_rows(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _write_preds(tmp_path, monkeypatch)
    f = tmp_path / "E1_logreg_tfidf.csv"
    d = pd.read_csv(f)
    d.drop(d.index[d["split"] == "test"][:1]).to_csv(f, index=False)
    with pytest.raises(ValueError, match="disagree"):
        report.load_split("test", tmp_path)


def test_significance_requires_all_e3_seeds(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _write_preds(tmp_path, monkeypatch)
    pairs = report.significance(report.load_split("test", tmp_path), n_boot=100)
    by = {(x["a"], x["b"]): x for x in pairs}
    assert len(by[("E2", "E3")]["tests"]) == 3  # one per seed, Holm across all 8 tests
    assert by[("E2", "E5")]["significant"] and by[("E2", "E5")]["better"] == "E2"
    assert by[("E4", "E5")]["tests"][0]["p_value"] == 1.0  # identical predictions


def test_full_report_writes_tables_json_and_figures(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _write_preds(tmp_path, monkeypatch)
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "E5_llm_few_shot.json").write_text(
        json.dumps({"latency_ms": {"p50": 190, "p95": 290, "mean": 180}, "device": "cuda", "cost_usd_per_1000": 0.0})
    )
    monkeypatch.setattr(report, "METRICS_DIR", metrics)
    monkeypatch.setattr(report, "FIG_DIR", tmp_path / "fig")
    monkeypatch.setattr(report, "TABLES_MD", tmp_path / "tables.md")
    monkeypatch.setattr(report, "REPO_ROOT", tmp_path)
    assert report.run(n_boot=50) == 0
    md = (tmp_path / "tables.md").read_text(encoding="utf-8")
    assert "| E2 XGBoost TF-IDF+SQL |" in md and "McNemar" in md and "$0.00" in md
    data = json.loads((metrics / "significance.json").read_text())
    assert set(data["pairs"]) == {"test", "external_test"}
    assert {p.name for p in (tmp_path / "fig").iterdir()} >= {"f1_ci.png", "latency_vs_f1.png", "confusion_E2_test.png"}
