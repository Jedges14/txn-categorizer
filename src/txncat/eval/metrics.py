"""Classification metrics (SPEC §7): accuracy, macro-F1, per-class F1, confusion matrix.

Macro-F1 averages over the classes PRESENT in y_true. That matters for the external test set, which covers 7 of the
13 categories: predicting one of the other 6 still counts as an error, but those 6 don't each add an F1 of 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


@dataclass(frozen=True)
class Scores:
    n: int
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    labels: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


Labels = Sequence[str] | np.ndarray


def score(y_true: Labels, y_pred: Labels) -> Scores:
    y_true_a, y_pred_a = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true_a) == 0:
        raise ValueError("no rows to score")
    labels = sorted(set(y_true_a.tolist()))
    per_class = f1_score(y_true_a, y_pred_a, labels=labels, average=None, zero_division=0)
    return Scores(
        n=len(y_true_a),
        accuracy=float(accuracy_score(y_true_a, y_pred_a)),
        macro_f1=float(np.mean(per_class)),
        per_class_f1={lab: float(f) for lab, f in zip(labels, per_class, strict=True)},
        labels=labels,
    )


def confusion(y_true: Labels, y_pred: Labels, labels: Sequence[str]) -> np.ndarray:
    """Rows = true class, columns = predicted class, in `labels` order (pass all 13 to see stray predictions)."""
    return confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=list(labels))
