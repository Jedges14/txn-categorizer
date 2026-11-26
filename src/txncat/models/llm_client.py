"""LLM clients for E4/E5 (SPEC §4: provider-agnostic `LLMClient`, disk cache so reruns cost nothing).

The implementation here is a FREE, open-weight model run locally with Hugging Face transformers on CPU (no API key,
no paid calls). Everything that varies per transaction is in the last user turn, so the shared prefix (instructions
and few-shot examples) is encoded once and its key/value cache reused for every row ("prompt-prefix caching").

Needs the optional `encoder` extra (torch + transformers):  uv sync --extra encoder
"""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from txncat.config import REPO_ROOT

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISIONS = {  # pinned for reproducibility; both Apache-2.0
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
}
CACHE_PATH = REPO_ROOT / ".cache" / "llm_responses.sqlite"

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": ...}


@dataclass(frozen=True)
class Completion:
    text: str
    latency_ms: float  # measured when the completion was first generated (cache hits keep the original)
    prompt_tokens: int
    output_tokens: int
    cached: bool


class LLMClient(Protocol):
    model_id: str

    def complete(self, prefix: list[Message], user: str) -> Completion:
        """Complete `prefix + [user turn]`; `prefix` is identical across calls and may be cached."""
        ...


class ResponseCache:
    """SQLite cache keyed by sha256(model, revision, generation settings, full prompt)."""

    def __init__(self, path: Path = CACHE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS completions (key TEXT PRIMARY KEY, text TEXT, latency_ms REAL, "
            "prompt_tokens INTEGER, output_tokens INTEGER)"
        )

    @staticmethod
    def key(model: str, revision: str | None, settings: dict, prefix: list[Message], user: str) -> str:
        payload = json.dumps([model, revision, settings, prefix, user], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Completion | None:
        row = self.db.execute(
            "SELECT text, latency_ms, prompt_tokens, output_tokens FROM completions WHERE key = ?", (key,)
        ).fetchone()
        return Completion(row[0], row[1], row[2], row[3], cached=True) if row else None

    def put(self, key: str, c: Completion) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO completions VALUES (?, ?, ?, ?, ?)",
            (key, c.text, c.latency_ms, c.prompt_tokens, c.output_tokens),
        )
        self.db.commit()


class LocalHFClient:
    """Greedy decoding with a local open-weight chat model; reuses the KV cache of the shared prompt prefix."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        revision: str | None = None,
        max_new_tokens: int = 16,
        cache: ResponseCache | None = None,
        threads: int | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self.model_id = model_id
        self.revision = revision if revision is not None else MODEL_REVISIONS.get(model_id)
        self.max_new_tokens = max_new_tokens
        self.cache = cache
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=self.revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=self.revision, dtype=torch.float32)
        self.model.eval()
        self._torch = torch
        self._prefix_key: str | None = None
        self._prefix_ids: list[int] = []
        self._prefix_kv = None

    @property
    def settings(self) -> dict:
        return {"decoding": "greedy", "max_new_tokens": self.max_new_tokens}

    def _ids(self, messages: list[Message], generation_prompt: bool) -> list[int]:
        out: Any = self.tok.apply_chat_template(messages, add_generation_prompt=generation_prompt, tokenize=True)
        # transformers 5 returns a BatchEncoding (dict with input_ids); older versions return a plain list of ids.
        ids = out["input_ids"] if hasattr(out, "keys") else out
        return [int(i) for i in ids]

    def _ensure_prefix(self, prefix: list[Message]) -> None:
        key = json.dumps(prefix, sort_keys=True)
        if key == self._prefix_key:
            return
        torch = self._torch
        ids = self._ids(prefix, generation_prompt=False)
        with torch.no_grad():
            out = self.model(torch.tensor([ids]), use_cache=True)
        self._prefix_key, self._prefix_ids, self._prefix_kv = key, ids, out.past_key_values

    def complete(self, prefix: list[Message], user: str) -> Completion:
        cache_key = ResponseCache.key(self.model_id, self.revision, self.settings, prefix, user)
        if self.cache and (hit := self.cache.get(cache_key)):
            return hit
        torch = self._torch
        self._ensure_prefix(prefix)
        full = self._ids([*prefix, {"role": "user", "content": user}], generation_prompt=True)
        reuse = full[: len(self._prefix_ids)] == self._prefix_ids  # chat templates are concatenative; verify
        started = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                input_ids=torch.tensor([full]),
                attention_mask=torch.ones(1, len(full), dtype=torch.long),
                past_key_values=copy.deepcopy(self._prefix_kv) if reuse else None,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.eos_token_id,
            )
        new_tokens = out[0, len(full) :]
        text = str(self.tok.decode(new_tokens, skip_special_tokens=True)).strip()
        c = Completion(text, (time.perf_counter() - started) * 1000, len(full), int(new_tokens.numel()), False)
        if self.cache:
            self.cache.put(cache_key, c)
        return c
