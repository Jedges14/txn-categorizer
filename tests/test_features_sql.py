"""SQL feature tests against real PostgreSQL (sql/features.sql)."""

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import OperationalError

from txncat.db import apply_schema, make_engine, server_version

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    eng = make_engine()
    try:
        server_version(eng)
    except OperationalError:
        if os.environ.get("REQUIRE_DB") == "1":
            raise
        pytest.skip("PostgreSQL not reachable at DATABASE_URL (run: docker compose up -d)")
    apply_schema(eng)
    yield eng
    eng.dispose()


# (raw description, expected desc_clean, expected merchant_token)
TRICKY = [
    ("SQ *BLUE BOTTLE COFFEE O", "BLUE BOTTLE COFFEE O", "BLUE BOTTLE"),  # Square prefix, truncated word
    ("TST* TACO BELL 0423 OAKLAND CA", "TACO BELL", "TACO BELL"),  # Toast prefix, store no., city + state
    ("CHECKCARD 0831 TRUE HAVEN", "TRUE HAVEN", "TRUE HAVEN"),  # card prefix with MMDD
    ("STARBUCKS #1234 SAN FRANCISCO CA", "STARBUCKS", "STARBUCKS"),  # #store and a two-word city
    ("76 12345 SALT LAKE CTY UT", "76", "76"),  # numeric brand kept; three-word city removed
    ("REFUND LITTLE CAR WASH", "LITTLE CAR WASH", "LITTLE CAR"),  # refund marker removed
    ("PADDLE.NET* FORMLOGIC.IO 2", "FORMLOGIC IO", "FORMLOGIC IO"),  # SaaS reseller, punctuation, trailing digit
    ("ACH DEBIT CAMDEN WATER DISTR", "ACH DEBIT CAMDEN WATER DISTR", "CAMDEN WATER"),  # channel kept, skipped in token
    ("UBER   *TRIP HELP.UBER.COM", "UBER TRIP HELP UBER COM", "UBER TRIP"),  # repeated spaces, * and dots
    ("FEDEX 789012345678 MEMPHIS TN", "FEDEX", "FEDEX"),  # long tracking number + city/state
    ("Mobile Deposit Ref 20250310", "MOBILE DEPOSIT REF", "MOBILE DEPOSIT"),  # mixed case, trailing date
    ("DOMINO'S 7721", "DOMINO'S", "DOMINO'S"),  # apostrophe kept
    ("AT&T BILL PAY", "AT&T BILL PAY", "AT&T BILL"),  # ampersand kept; "PAY" isn't a state
]


@pytest.mark.parametrize(("raw", "clean", "token"), TRICKY)
def test_clean_desc_and_merchant_token(engine: Engine, raw: str, clean: str, token: str) -> None:
    with engine.connect() as conn:
        got_clean, got_token = conn.execute(
            text("SELECT txncat_clean_desc(:r), txncat_merchant_token(txncat_clean_desc(:r))"), {"r": raw}
        ).one()
    assert (got_clean, got_token) == (clean, token)


def test_clean_desc_handles_null_and_empty(engine: Engine) -> None:
    with engine.connect() as conn:
        assert conn.execute(text("SELECT txncat_clean_desc(NULL), txncat_clean_desc('')")).one() == ("", "")


def test_train_only_statistics_ignore_other_splits(engine: Engine) -> None:
    """Adding non-train rows must not move the amount cut points or create merchant frequency."""
    with engine.connect() as conn, conn.begin() as tx:
        before = conn.execute(text("SELECT cuts FROM train_amount_cutpoints")).scalar_one()
        conn.execute(
            text(
                "INSERT INTO transactions (posted_at, description, amount, category, split, source) "
                "SELECT now(), 'ZZQX LEAKCHECK SHOP', -999999.00, 'Other', s, 'test-leak' "
                "FROM unnest(ARRAY['val','test','test','test']) AS s"
            )
        )
        after = conn.execute(text("SELECT cuts FROM train_amount_cutpoints")).scalar_one()
        freq = conn.execute(
            text("SELECT merchant_freq FROM txn_features WHERE source = 'test-leak' LIMIT 1")
        ).scalar_one()
        tx.rollback()
    assert before == after
    assert freq == 0


def test_feature_view_semantics(engine: Engine) -> None:
    with engine.connect() as conn, conn.begin() as tx:
        conn.execute(
            text(
                "INSERT INTO transactions (posted_at, description, amount, direction, category, split, source) VALUES "
                "('2025-03-01', 'REFUND SQ *JOE CAFE', 12.00, NULL, 'Meals & Entertainment', 'test', 'test-sem'), "
                "(NULL, 'Zelle payment from someone', NULL, 'credit', 'Income / Deposits', 'external_test', 'test-sem')"
            )
        )
        rows = conn.execute(
            text(
                "SELECT description, is_outflow, is_refund, has_processor_prefix, day_of_week, is_weekend, "
                "amount_bucket FROM txn_features WHERE source = 'test-sem' ORDER BY description"
            )
        ).all()
        tx.rollback()
    refund, ext = rows
    assert refund.is_outflow is False and refund.is_refund is True and refund.has_processor_prefix is True
    assert refund.day_of_week == 6 and refund.is_weekend is True  # 2025-03-01 was a Saturday
    assert refund.amount_bucket is not None
    assert ext.is_outflow is False  # from direction='credit' when there is no amount
    assert ext.day_of_week is None and ext.amount_bucket is None
