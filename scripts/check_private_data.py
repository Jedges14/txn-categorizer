"""Pre-commit guard (CLAUDE.md rule): data/private/ must stay git-ignored and never be committed.

Fails if .gitignore stops ignoring data/private/, or if any file under it is staged or tracked.
"""

from __future__ import annotations

import subprocess
import sys

PRIVATE = "data/private/"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def main() -> int:
    problems: list[str] = []
    probe = f"{PRIVATE}__probe__.csv"
    ignored = subprocess.run(["git", "check-ignore", "-q", probe], check=False).returncode == 0
    if not ignored:
        problems.append(f"{PRIVATE} is not ignored by .gitignore")
    tracked = [p for p in git("ls-files").splitlines() if p.startswith(PRIVATE)]
    staged = [p for p in git("diff", "--cached", "--name-only").splitlines() if p.startswith(PRIVATE)]
    for p in sorted(set(tracked + staged)):
        problems.append(f"private file is tracked or staged: {p}")
    for msg in problems:
        print(f"check-private-data: {msg}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
