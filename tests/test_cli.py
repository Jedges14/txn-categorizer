import time

import pytest

from txncat import __version__
from txncat.cli import _redact, build_parser, main


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"txncat {__version__}"


def test_subcommand_is_required() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "postgresql+psycopg://txncat:secret@localhost:5433/txncat",
            "postgresql+psycopg://txncat:***@localhost:5433/txncat",
        ),
        ("postgresql://u:p@ss@word@host/db", "postgresql://u:***@host/db"),
        ("sqlite:///local.db", "sqlite:///local.db"),
    ],
)
def test_redact_hides_password(url: str, expected: str) -> None:
    assert _redact(url) == expected


def test_db_check_reports_unreachable_database(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Port 1 on localhost: nothing listens. The connect timeout keeps this from hanging.
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:hunter2@127.0.0.1:1/x")
    monkeypatch.setenv("DB_CONNECT_TIMEOUT_S", "2")
    start = time.monotonic()
    assert main(["db", "check"]) == 1
    assert time.monotonic() - start < 15, "db check must give up quickly when Postgres is down"
    err = capsys.readouterr().err
    assert "Cannot reach PostgreSQL" in err
    assert "hunter2" not in err
