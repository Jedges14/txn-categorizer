"""Guards for the project rules in CLAUDE.md that can be checked automatically."""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _ignored(path: str) -> bool:
    return (
        subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO, check=False).returncode == 0
    )


def test_private_data_and_env_files_are_git_ignored() -> None:
    assert _ignored("data/private/real_test.csv")
    assert _ignored(".env")
    assert not _ignored(".env.example")


def test_no_private_files_are_tracked() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert not [p for p in tracked if p.startswith("data/private/")]
    assert ".env" not in tracked


def test_env_example_has_no_real_keys() -> None:
    for line in (REPO / ".env.example").read_text(encoding="utf-8").splitlines():
        if line.startswith(("ANTHROPIC_API_KEY=", "OPENAI_API_KEY=")):
            assert line.split("=", 1)[1] == "", f"placeholder must be empty: {line.split('=')[0]}"
