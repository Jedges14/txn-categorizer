"""E4/E5 tests with a fake LLM client: prompt building, few-shot leakage guard, label parsing, cache. No model
download, no network, no paid calls."""

from pathlib import Path

import pandas as pd
import pytest

from txncat.data.taxonomy import category_names
from txncat.models.llm_client import Completion, ResponseCache
from txncat.models.llm_fewshot import (
    INVALID,
    build_prefix,
    few_shot_examples,
    parse_label,
    run_llm,
    system_prompt,
    user_turn,
)

LABELS = category_names()


def _frame() -> pd.DataFrame:
    rows = []
    i = 0
    for split in ("train", "val", "test"):
        for cat in LABELS:
            for j in range(5):
                rows.append(
                    {
                        "id": i,
                        "split": split,
                        "category": cat,
                        "description": f"{cat.upper()} {j}",
                        "amount": -10.0 - j,
                        "direction": None,
                    }
                )
                i += 1
    return pd.DataFrame(rows)


class FakeClient:
    """Answers with the true category for 'train-like' text, a messy variant, or garbage, to exercise parsing."""

    model_id = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[list, str]] = []

    def complete(self, prefix: list, user: str) -> Completion:
        self.calls.append((prefix, user))
        desc = user.splitlines()[0].removeprefix("Description: ")
        cat = next(c for c in LABELS if desc.startswith(c.upper()))
        reply = {"0": cat, "1": f"{cat.lower()}.", "2": f"Category: {cat}"}.get(desc[-1], "no idea")
        return Completion(reply, 12.0, 100, 4, cached=False)


def test_system_prompt_lists_every_category_with_definition() -> None:
    sp = system_prompt()
    for name in LABELS:
        assert f"- {name}: " in sp
    assert "{categories}" not in sp


@pytest.mark.parametrize(
    ("amount", "direction", "expected"),
    [
        (-12.5, None, "Amount: 12.50\nDirection: money out"),
        (40.0, None, "Amount: 40.00\nDirection: money in"),
        (float("nan"), "credit", "Amount: unknown\nDirection: money in"),
        (float("nan"), None, "Amount: unknown\nDirection: unknown"),
    ],
)
def test_user_turn(amount: float, direction: str | None, expected: str) -> None:
    assert user_turn(pd.Series({"description": "X", "amount": amount, "direction": direction})).endswith(expected)


def test_few_shot_examples_are_train_only_seeded_and_k_per_class() -> None:
    df = _frame()
    train = df[df["split"] == "train"]
    ex = few_shot_examples(train)
    assert len(ex) == 3 * len(LABELS)
    assert ex["category"].value_counts().eq(3).all()
    assert ex["id"].tolist() == few_shot_examples(train)["id"].tolist()  # deterministic
    with pytest.raises(ValueError, match="train split only"):
        few_shot_examples(df[df["split"].isin(["train", "test"])])


def test_build_prefix_modes() -> None:
    zero, none = build_prefix("zero")
    assert [m["role"] for m in zero] == ["system"] and none == []
    few, ids = build_prefix("few", _frame().query("split == 'train'"))
    assert len(few) == 1 + 2 * 39 and len(ids) == 39
    assert [m["role"] for m in few[1:3]] == ["user", "assistant"]
    with pytest.raises(ValueError):
        build_prefix("three-shot")


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Travel", "Travel"),
        ("  meals and entertainment. ", "Meals & Entertainment"),
        ("Income/Deposits", "Income / Deposits"),
        ("Category: Fuel & Auto", "Fuel & Auto"),
        ("Bank Fees & Charges\nbecause it says FEE", "Bank Fees & Charges"),
        ("Office Supplies or Software & Subscriptions", INVALID),  # ambiguous
        ("I am not sure", INVALID),
        ("", INVALID),
    ],
)
def test_parse_label(reply: str, expected: str) -> None:
    assert parse_label(reply, LABELS) == expected


def test_run_llm_counts_invalid_and_never_drops_rows() -> None:
    df = _frame()
    client = FakeClient()
    r = run_llm(df, client, "few", ("test",), log=lambda _: None)
    assert len(r.predictions) == (df["split"] == "test").sum()  # every row kept
    invalid = (r.predictions["pred"] == INVALID).sum()
    assert invalid == 2 * len(LABELS)  # descriptions ending in 3 and 4 get "no idea"
    assert len(r.example_ids) == 39 and set(r.example_ids) <= set(df.loc[df["split"] == "train", "id"])
    # The shared prefix is identical on every call, so a real client can cache it.
    assert all(p == client.calls[0][0] for p, _ in client.calls)


def test_response_cache_roundtrip(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "c.sqlite")
    key = ResponseCache.key("m", "rev", {"a": 1}, [{"role": "system", "content": "s"}], "u")
    assert cache.get(key) is None
    cache.put(key, Completion("Travel", 50.0, 10, 2, cached=False))
    hit = cache.get(key)
    assert hit is not None and hit.text == "Travel" and hit.cached and hit.latency_ms == 50.0
    assert key != ResponseCache.key("m", "rev", {"a": 2}, [{"role": "system", "content": "s"}], "u")
