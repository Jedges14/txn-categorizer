"""E4 (zero-shot) and E5 (few-shot) LLM classification (SPEC §7).

Prompt = instructions + category definitions (+ for E5, k examples per category from TRAIN, fixed seed) + one user turn
per transaction. Replies are mapped to a category by `parse_label`; anything that can't be mapped is INVALID, counted
and reported, and scored as wrong (never silently dropped).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from txncat.config import REPO_ROOT
from txncat.data.taxonomy import load_taxonomy
from txncat.models.llm_client import LLMClient, Message

PROMPT_PATH = REPO_ROOT / "prompts" / "system.md"
INVALID = "INVALID"
FEW_SHOT_K = 3
FEW_SHOT_SEED = 42


def system_prompt() -> str:
    cats = "\n".join(f"- {c.name}: {c.definition}" for c in load_taxonomy())
    return PROMPT_PATH.read_text(encoding="utf-8").format(categories=cats).strip()


def user_turn(row: pd.Series) -> str:
    amount = row.get("amount")
    known = amount is not None and not (isinstance(amount, float) and pd.isna(amount))
    if known:
        direction = "money out" if float(amount) < 0 else "money in"
        amount_txt = f"{abs(float(amount)):.2f}"
    else:
        d = row.get("direction")
        direction = {"debit": "money out", "credit": "money in"}.get(d, "unknown") if isinstance(d, str) else "unknown"
        amount_txt = "unknown"
    return f"Description: {row['description']}\nAmount: {amount_txt}\nDirection: {direction}"


def few_shot_examples(train: pd.DataFrame, k: int = FEW_SHOT_K, seed: int = FEW_SHOT_SEED) -> pd.DataFrame:
    """k rows per category from TRAIN, chosen and shuffled with a fixed seed."""
    if set(train["split"]) - {"train"}:
        raise ValueError("few-shot examples must come from the train split only")
    rng = random.Random(seed)
    picks: list[int] = []
    for cat in sorted(train["category"].unique()):
        ids = sorted(train.loc[train["category"] == cat, "id"].tolist())
        picks.extend(rng.sample(ids, min(k, len(ids))))
    rng.shuffle(picks)
    return train.set_index("id").loc[picks].reset_index()


def build_prefix(mode: str, train: pd.DataFrame | None = None) -> tuple[list[Message], list[int]]:
    prefix: list[Message] = [{"role": "system", "content": system_prompt()}]
    example_ids: list[int] = []
    if mode == "few":
        if train is None:
            raise ValueError("few-shot mode needs train rows")
        ex = few_shot_examples(train)
        for _, r in ex.iterrows():
            prefix += [{"role": "user", "content": user_turn(r)}, {"role": "assistant", "content": r["category"]}]
        example_ids = [int(i) for i in ex["id"]]
    elif mode != "zero":
        raise ValueError(f"mode must be 'zero' or 'few', not {mode!r}")
    return prefix, example_ids


def _norm(s: str) -> str:
    s = s.lower().replace(" and ", " & ")
    s = re.sub(r"\s*([&/])\s*", r" \1 ", s)
    return re.sub(r"[^a-z&/ ]", "", s).strip()


def parse_label(text: str, labels: list[str]) -> str:
    """Map the first line of a reply to a category: an exact (normalized) match, else a line that names exactly one
    category (e.g. "Category: Travel"). A line naming two or more categories is ambiguous: INVALID, never a guess."""
    reply = _norm(text.strip().splitlines()[0] if text.strip() else "")
    if not reply:
        return INVALID
    by_norm = {_norm(lab): lab for lab in labels}
    if reply in by_norm:
        return by_norm[reply]
    contained = [lab for n, lab in by_norm.items() if n in reply]
    return contained[0] if len(contained) == 1 else INVALID


@dataclass
class LLMRun:
    mode: str
    predictions: pd.DataFrame  # id, split, category, pred, raw, latency_ms, cached
    example_ids: list[int]
    prompt_sha256: str
    seconds: float


def run_llm(
    df: pd.DataFrame,
    client: LLMClient,
    mode: str,
    splits: tuple[str, ...],
    limit_per_split: int | None = None,
    log: Callable[[str], None] = print,
) -> LLMRun:
    labels = [c.name for c in load_taxonomy()]
    prefix, example_ids = build_prefix(mode, df[df["split"] == "train"] if mode == "few" else None)
    prompt_sha = hashlib.sha256(json.dumps(prefix, ensure_ascii=False).encode("utf-8")).hexdigest()
    rows = []
    started = time.perf_counter()
    for split in splits:
        part = df[df["split"] == split].sort_values("id")
        if limit_per_split is not None:
            part = part.sample(n=min(limit_per_split, len(part)), random_state=0).sort_values("id")
        for i, (_, r) in enumerate(part.iterrows(), start=1):
            c = client.complete(prefix, user_turn(r))
            rows.append(
                {
                    "id": int(r["id"]),
                    "split": split,
                    "category": r["category"],
                    "pred": parse_label(c.text, labels),
                    "raw": c.text,
                    "latency_ms": round(c.latency_ms, 1),
                    "cached": c.cached,
                }
            )
            if i % 250 == 0 or i == len(part):
                recent = [x["latency_ms"] for x in rows[-250:]]
                log(
                    f"E{'4' if mode == 'zero' else '5'} {split} {i}/{len(part)} "
                    f"median latency {statistics.median(recent):.0f} ms"
                )
    return LLMRun(mode, pd.DataFrame(rows), example_ids, prompt_sha, time.perf_counter() - started)
