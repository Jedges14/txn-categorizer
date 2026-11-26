"""E3: fine-tuned small transformer encoder (SPEC §7, M3).

DistilBERT (uncased) + a classification head, fine-tuned on desc_clean (the same text E1/E2 see; text only, no
amount/date, so it can't fail the way E2 does when those are missing). Class-weighted cross-entropy (like E1/E2's
balanced weighting). After every epoch the model is scored on VAL; the best epoch's weights are kept. Test and
external sets are predicted once, with those weights. Run per seed; results are reported as mean ± std over seeds.

Needs the optional `encoder` extra:  uv sync --extra encoder
"""

from __future__ import annotations

import copy
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from txncat.data.taxonomy import category_names
from txncat.eval.metrics import score

BASE_MODEL = "distilbert/distilbert-base-uncased"
BASE_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"  # pinned for reproducibility (Apache-2.0)


@dataclass
class FinetuneConfig:
    model_name: str = BASE_MODEL
    revision: str | None = BASE_REVISION
    max_length: int = 32  # descriptions are ≤32 chars synthetic, longer externally; 32 tokens covers nearly all
    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 0
    class_weighted: bool = True
    max_steps: int | None = None  # smoke tests only
    device: str | None = None  # default: cuda if available, else cpu


@dataclass
class SeedResult:
    seed: int
    val_macro_f1_by_epoch: list[float]
    best_epoch: int
    predictions: pd.DataFrame  # id, split, category, pred
    seconds: float
    device: str


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batches(n: int, size: int, shuffle_rng: np.random.Generator | None) -> list[np.ndarray]:
    idx = np.arange(n)
    if shuffle_rng is not None:
        shuffle_rng.shuffle(idx)
    return [idx[i : i + size] for i in range(0, n, size)]


def train_one_seed(
    df: pd.DataFrame,
    cfg: FinetuneConfig,
    eval_splits: tuple[str, ...] = ("val", "test", "external_test"),
    log: Callable[[str], None] = print,
) -> SeedResult:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    _set_seed(cfg.seed)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    labels = category_names()
    index = {lab: i for i, lab in enumerate(labels)}

    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    evals = df[df["split"].isin(eval_splits)].reset_index(drop=True)
    if train.empty or val.empty:
        raise ValueError("need train and val rows")

    tok = AutoTokenizer.from_pretrained(cfg.model_name, revision=cfg.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, revision=cfg.revision, num_labels=len(labels)
    ).to(device)

    def encode(texts: pd.Series) -> dict:
        return tok(
            texts.fillna("").tolist(), padding=True, truncation=True, max_length=cfg.max_length, return_tensors="pt"
        )

    y_train = torch.tensor(train["category"].map(index).to_numpy(), dtype=torch.long)
    counts = np.bincount(y_train.numpy(), minlength=len(labels)).astype("float64")
    weights = None
    if cfg.class_weighted:
        w = np.where(counts > 0, len(y_train) / (len(labels) * np.maximum(counts, 1)), 0.0)
        weights = torch.tensor(w, dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=cfg.learning_rate)
    steps_per_epoch = -(-len(train) // cfg.batch_size)
    total_steps = cfg.max_steps or steps_per_epoch * cfg.epochs
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total_steps), total_steps)

    @torch.no_grad()
    def predict(frame: pd.DataFrame) -> np.ndarray:
        model.eval()
        out: list[np.ndarray] = []
        for b in _batches(len(frame), 256, None):
            enc = {k: v.to(device) for k, v in encode(frame["desc_clean"].iloc[b]).items()}
            out.append(model(**enc).logits.argmax(-1).cpu().numpy())
        return np.asarray(labels)[np.concatenate(out)] if out else np.asarray([], dtype=object)

    rng = np.random.default_rng(cfg.seed)
    started = time.perf_counter()
    val_f1: list[float] = []
    best_state, best_f1, best_epoch, step = None, -1.0, 0, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        e0 = time.perf_counter()
        for b in _batches(len(train), cfg.batch_size, rng):
            enc = {k: v.to(device) for k, v in encode(train["desc_clean"].iloc[b]).items()}
            loss = loss_fn(model(**enc).logits, y_train[b].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if cfg.max_steps and step >= cfg.max_steps:
                break
        f1 = score(val["category"], predict(val)).macro_f1
        val_f1.append(f1)
        log(f"E3 seed={cfg.seed} epoch {epoch}/{cfg.epochs} val macro-F1={f1:.4f} ({time.perf_counter() - e0:.0f}s)")
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        if cfg.max_steps and step >= cfg.max_steps:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    preds = predict(evals)
    frame = pd.DataFrame(
        {
            "id": evals["id"].to_numpy(),
            "split": evals["split"].to_numpy(),
            "category": evals["category"].to_numpy(),
            "pred": preds,
        }
    )
    return SeedResult(cfg.seed, val_f1, best_epoch, frame, time.perf_counter() - started, device)


def config_dict(cfg: FinetuneConfig) -> dict:
    return asdict(cfg)
