"""The fixed category taxonomy (data/taxonomy.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

from txncat.config import REPO_ROOT

TAXONOMY_PATH = REPO_ROOT / "data" / "taxonomy.yaml"


@dataclass(frozen=True)
class Category:
    name: str
    definition: str
    examples: tuple[str, ...]


@cache
def load_taxonomy(path: Path = TAXONOMY_PATH) -> tuple[Category, ...]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return tuple(Category(c["name"], c["definition"], tuple(c["examples"])) for c in raw["categories"])


def category_names(path: Path = TAXONOMY_PATH) -> list[str]:
    return [c.name for c in load_taxonomy(path)]
