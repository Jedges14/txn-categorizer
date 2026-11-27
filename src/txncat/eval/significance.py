"""Statistical comparison of classifiers evaluated on the SAME rows (SPEC §7, M5).

- McNemar's test (exact binomial on the discordant pairs) for "does model A make different mistakes than B?"
- Holm's step-down correction when several pairs are tested together.
- Bootstrap 95% CIs (1,000 resamples of rows) for macro-F1, and a paired bootstrap CI for the difference A − B.
Macro-F1 always averages over the classes present in the FULL y_true, so resamples are comparable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from txncat.eval.metrics import Labels

N_BOOT = 1000
SEED = 0


@dataclass(frozen=True)
class McNemar:
    n: int
    a_right_b_wrong: int  # n10
    a_wrong_b_right: int  # n01
    p_value: float
    method: str = "exact binomial (two-sided) on discordant pairs"

    def as_dict(self) -> dict:
        return asdict(self)


def mcnemar(y: Labels, a: Labels, b: Labels) -> McNemar:
    y_, a_, b_ = np.asarray(y), np.asarray(a), np.asarray(b)
    if not (len(y_) == len(a_) == len(b_)):
        raise ValueError("y, a and b must be aligned on the same rows")
    a_ok, b_ok = a_ == y_, b_ == y_
    n10 = int(np.sum(a_ok & ~b_ok))
    n01 = int(np.sum(~a_ok & b_ok))
    p = 1.0 if n10 + n01 == 0 else float(binomtest(n10, n10 + n01, 0.5).pvalue)
    return McNemar(len(y_), n10, n01, p)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (same order as given), monotone and capped at 1."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def macro_f1(y: Labels, pred: Labels) -> float:
    """Reference implementation (sklearn), averaged over the classes present in y."""
    y_ = np.asarray(y)
    return float(f1_score(y_, np.asarray(pred), labels=sorted(set(y_.tolist())), average="macro", zero_division=0))


def _encode(y: np.ndarray, *preds: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], int]:
    """Integer codes: classes of y are 0..k-1; any prediction outside them (e.g. INVALID) gets code k."""
    labels = sorted(set(y.tolist()))
    index = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    return np.array([index[v] for v in y]), [np.array([index.get(v, k) for v in p]) for p in preds], k


def _boot_f1(yc: np.ndarray, pc: np.ndarray, k: int, idx: np.ndarray) -> np.ndarray:
    """Macro-F1 for every resample at once. Rows of idx are resamples. F1_c = 2TP / (2TP + FP + FN)."""
    out = np.empty(len(idx))
    for r, i in enumerate(idx):
        yt, pt = yc[i], pc[i]
        tp = np.bincount(yt[yt == pt], minlength=k + 1)[:k]
        support = np.bincount(yt, minlength=k + 1)[:k]
        predicted = np.bincount(pt, minlength=k + 1)[:k]
        denom = support + predicted
        f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
        out[r] = f1.mean()
    return out


def _indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_macro_f1(y: Labels, pred: Labels, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """(point estimate, 2.5th percentile, 97.5th percentile) of macro-F1 over row resamples.
    Classes are fixed to those in the full y, so a class missing from a resample scores F1 = 0 for it."""
    y_, p_ = np.asarray(y), np.asarray(pred)
    yc, (pc,), k = _encode(y_, p_)
    stats = _boot_f1(yc, pc, k, _indices(len(y_), n_boot, seed))
    return macro_f1(y_, p_), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def bootstrap_diff(
    y: Labels, a: Labels, b: Labels, n_boot: int = N_BOOT, seed: int = SEED
) -> tuple[float, float, float]:
    """Paired bootstrap of macro-F1(a) − macro-F1(b): the same resampled rows for both models."""
    y_, a_, b_ = np.asarray(y), np.asarray(a), np.asarray(b)
    yc, (ac, bc), k = _encode(y_, a_, b_)
    idx = _indices(len(y_), n_boot, seed)
    diffs = _boot_f1(yc, ac, k, idx) - _boot_f1(yc, bc, k, idx)
    point = macro_f1(y_, a_) - macro_f1(y_, b_)
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
