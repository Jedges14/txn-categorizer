"""TF-IDF text features on desc_clean (SPEC §6): word 1–2 grams + character 3–5 grams. Fit on TRAIN only."""

from __future__ import annotations

from collections.abc import Iterable

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def make_tfidf(max_word: int = 20_000, max_char: int = 30_000) -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    token_pattern=r"(?u)\b[\w&']+\b",
                    min_df=2,
                    max_features=max_word,
                    sublinear_tf=True,
                    lowercase=False,  # desc_clean is already upper-case
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=max_char,
                    sublinear_tf=True,
                    lowercase=False,
                ),
            ),
        ]
    )


def fit_transform_train(texts: Iterable[str]) -> tuple[FeatureUnion, sparse.csr_matrix]:
    vec = make_tfidf()
    X = vec.fit_transform(list(texts))
    return vec, sparse.csr_matrix(X)
