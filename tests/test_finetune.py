"""E3 smoke test (SPEC M3): tiny subset, 1 training step, CPU, fully offline.

Uses a tiny randomly initialised DistilBERT and a WordPiece tokenizer built from the test's own texts, so CI never
downloads the real model. Skipped if the optional `encoder` extra isn't installed.
"""

from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from txncat.data.generate_synthetic import generate  # noqa: E402
from txncat.data.merchants import build_merchants  # noqa: E402
from txncat.data.taxonomy import category_names  # noqa: E402
from txncat.models.finetune import FinetuneConfig, train_one_seed  # noqa: E402


def _tiny_model(texts: list[str], out: Path) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import DistilBertConfig, DistilBertForSequenceClassification, PreTrainedTokenizerFast

    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    raw = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    raw.train_from_iterator(texts, trainers.WordPieceTrainer(vocab_size=300, special_tokens=specials))
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    cfg = DistilBertConfig(
        vocab_size=tok.vocab_size,
        dim=32,
        n_layers=1,
        n_heads=2,
        hidden_dim=64,
        max_position_embeddings=64,
        num_labels=len(category_names()),
        pad_token_id=tok.pad_token_id,
    )
    torch.manual_seed(0)
    DistilBertForSequenceClassification(cfg).save_pretrained(out)
    tok.save_pretrained(out)
    return out


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rows = generate(build_merchants(), n_rows=600, seed=3)
    return pd.DataFrame(
        {
            "id": range(len(rows)),
            "split": [t.split for t in rows],
            "category": [t.category for t in rows],
            "description": [t.description for t in rows],
            "desc_clean": [t.description for t in rows],
        }
    )


def test_finetune_smoke_one_step(frame: pd.DataFrame, tmp_path: Path) -> None:
    model_dir = _tiny_model(frame["desc_clean"].tolist(), tmp_path / "tiny")
    cfg = FinetuneConfig(model_name=str(model_dir), revision=None, epochs=1, max_steps=1, batch_size=8, device="cpu")
    logs: list[str] = []
    result = train_one_seed(frame, cfg, eval_splits=("val", "test"), log=logs.append)
    n_eval = int(frame["split"].isin(["val", "test"]).sum())
    assert len(result.predictions) == n_eval
    assert set(result.predictions["pred"]) <= set(category_names())
    assert result.best_epoch == 1 and len(result.val_macro_f1_by_epoch) == 1
    assert logs and "val macro-F1" in logs[0]


def test_finetune_is_seeded(frame: pd.DataFrame, tmp_path: Path) -> None:
    model_dir = _tiny_model(frame["desc_clean"].tolist(), tmp_path / "tiny")
    cfg = FinetuneConfig(model_name=str(model_dir), revision=None, epochs=1, max_steps=3, batch_size=8, device="cpu")
    a = train_one_seed(frame, cfg, eval_splits=("val",), log=lambda _: None).predictions
    b = train_one_seed(frame, cfg, eval_splits=("val",), log=lambda _: None).predictions
    pd.testing.assert_frame_equal(a, b)
