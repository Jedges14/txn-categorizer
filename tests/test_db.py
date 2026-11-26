"""Schema tests against a real PostgreSQL (docker compose locally, a service container in CI).

Skipped, not failed, when DATABASE_URL is unreachable, so `uv run pytest` works without Docker.
"""

import os
from collections.abc import Iterator
from decimal import Decimal

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError, OperationalError

from txncat.db import apply_schema, make_engine, server_version

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    eng = make_engine()
    try:
        server_version(eng)
    except OperationalError:
        if os.environ.get("REQUIRE_DB") == "1":  # CI: an unreachable DB is a failure, not a skip
            raise
        pytest.skip("PostgreSQL not reachable at DATABASE_URL (run: docker compose up -d)")
    yield eng
    eng.dispose()


def test_postgres_16(engine: Engine) -> None:
    assert server_version(engine).startswith("16.")


def test_schema_is_idempotent(engine: Engine) -> None:
    apply_schema(engine)
    apply_schema(engine)  # second run must not fail
    with engine.connect() as conn:
        cols = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'transactions' ORDER BY ordinal_position"
            )
        ).scalars()
        assert list(cols) == [
            "id",
            "posted_at",
            "description",
            "amount",
            "category",
            "split",
            "source",
        ]


def _insert(conn, split: str | None, amount: str = "-4.75"):  # type: ignore[no-untyped-def]
    return conn.execute(
        text(
            "INSERT INTO transactions (posted_at, description, amount, category, split, source) "
            "VALUES (now(), 'SQ *BLUE BOTTLE 0423 OAKLAND CA', :amount, 'Meals & Entertainment', "
            ":split, 'synthetic') RETURNING amount"
        ),
        {"split": split, "amount": amount},
    ).scalar_one()


def test_valid_rows_insert_and_amount_keeps_cents(engine: Engine) -> None:
    apply_schema(engine)
    with engine.connect() as conn, conn.begin() as tx:
        for split in ("train", "val", "test", "real_test", None):
            assert _insert(conn, split) == Decimal("-4.75")
        tx.rollback()  # leave the table as we found it


def test_load_replaces_rows_per_source_without_duplicates(engine: Engine) -> None:
    from txncat.config import REPO_ROOT
    from txncat.data.generate_synthetic import generate
    from txncat.data.load_to_postgres import read_real_test_csv, replace_source
    from txncat.data.merchants import build_merchants

    apply_schema(engine)
    synthetic = [
        {
            "posted_at": t.posted_at,
            "description": t.description,
            "amount": t.amount,
            "category": t.category,
            "split": t.split,
            "source": "test-synthetic",
        }
        for t in generate(build_merchants(), n_rows=300, seed=1)
    ]
    real = [
        dict(r, source="test-real") for r in read_real_test_csv(REPO_ROOT / "data" / "sample" / "real_test_example.csv")
    ]
    try:
        first = replace_source(engine, "test-synthetic", synthetic)
        again = replace_source(engine, "test-synthetic", synthetic)  # rerun must replace, not append
        assert first.rows == again.rows == 300
        assert sum(again.by_split.values()) == 300
        assert set(again.by_split) <= {"train", "val", "test"}
        r = replace_source(engine, "test-real", real)
        assert r.by_split == {"real_test": 20}
    finally:  # leave the dev database as we found it
        replace_source(engine, "test-synthetic", [])
        replace_source(engine, "test-real", [])


def test_split_check_constraint_rejects_unknown_split(engine: Engine) -> None:
    apply_schema(engine)
    with engine.connect() as conn, conn.begin() as tx:
        with pytest.raises(IntegrityError):
            _insert(conn, "holdout")
        tx.rollback()
