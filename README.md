# Transaction Categorizer: classical ML vs. LLM
n[![CI](https://github.com/Jedges14/txn-categorizer/actions/workflows/ci.yml/badge.svg)](https://github.com/Jedges14/txn-categorizer/actions/workflows/ci.yml)

Sorts short bank-transaction descriptions (e.g. `SQ *BLUE BOTTLE 0423 OAKLAND CA`) into small-business expense
categories, and compares an XGBoost baseline on SQL-engineered features, a fine-tuned small transformer, and a
few-shot LLM on accuracy, macro-F1, statistical significance, cost and latency. Plan: [SPEC.md](SPEC.md).

> Status: **M0 (setup) done**; CI green on `2db2c33`. No data, models or results yet; every number will be recorded in `RESULTS.md` with the
> command that produced it.

## Run locally

Requires Python 3.11–3.13, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
docker compose up -d          # PostgreSQL 16 on localhost:5433 (schema created on first start)
uv sync                       # create .venv with pinned dependencies
uv run txncat db check        # confirm the database is reachable
uv run pytest                 # tests (DB tests skip if Postgres isn't running)
uv run pre-commit install     # run ruff + repo checks on every commit
```

Copy `.env.example` to `.env` only if you need to change the database URL or (from M4) add an LLM key.

## Layout

```
src/txncat/   cli.py (txncat), config.py, db.py; data/ features/ models/ eval/ filled in M1–M5
sql/          001_schema.sql (transactions table); features.sql arrives in M2
data/         sample/ (fake rows, committed) · private/ (your real labeled transactions; git-ignored, never committed)
tests/        CLI, repo-hygiene guards, schema tests against real Postgres
```

## Data privacy

`data/private/` is git-ignored, a pre-commit hook and a test fail if anything there is ever staged or tracked, and
CI never calls paid APIs.
