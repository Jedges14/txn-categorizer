# Transaction Categorizer: classical ML vs. LLM

[![CI](https://github.com/Jedges14/txn-categorizer/actions/workflows/ci.yml/badge.svg)](https://github.com/Jedges14/txn-categorizer/actions/workflows/ci.yml)

Sorts short bank-transaction descriptions (e.g. `SQ *BLUE BOTTLE 0423 OAKLAND CA`) into small-business expense
categories, and compares an XGBoost baseline on SQL-engineered features, a fine-tuned small transformer, and a
few-shot LLM on accuracy, macro-F1, statistical significance, cost and latency. Plan: [SPEC.md](SPEC.md).

> Status: **M1 (data) done**: taxonomy, 20,000 synthetic transactions split by merchant, plus a 2,100-row external
> test set from an independent dataset, all in Postgres. There is no real-transaction test set; see
> [data/README.md](data/README.md).
> No models or results yet; every number is recorded in [RESULTS.md](RESULTS.md) with the command that produced it.

![Class distribution](reports/figures/class_distribution.png)

## Run locally

Requires Python 3.12–3.13, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
docker compose up -d          # PostgreSQL 16 on localhost:5433 (schema created on first start)
uv sync                       # create .venv with pinned dependencies
uv run txncat db check        # confirm the database is reachable
uv run txncat data build      # data/merchants.csv + data/generated/synthetic.csv (seed 42)
uv run txncat data external   # download the pinned external test dataset (MIT, checksum-verified)
uv run txncat data load       # into Postgres (synthetic + external test; + private real test if present)
uv run txncat data report     # reports/data_summary.md + reports/figures/class_distribution.png
uv run pytest                 # tests (DB tests skip if Postgres isn't running)
uv run pre-commit install     # run ruff + repo checks on every commit
```

Copy `.env.example` to `.env` only if you need to change the database URL or (from M4) add an LLM key.

## Layout

```
src/txncat/   cli.py (txncat), config.py, db.py; data/ (M1); features/ models/ eval/ filled in M2–M5
sql/          001_schema.sql (transactions table); features.sql arrives in M2
data/         taxonomy.yaml, merchants.csv, sample/ (fake rows) · private/ (your real labels; git-ignored)
              how the data was made and its limitations: data/README.md
notebooks/    01_eda.ipynb (executed)
tests/        CLI, data/splits (no merchant in two splits), repo hygiene, schema + loading against real Postgres
```

## Data privacy

`data/private/` is git-ignored, a pre-commit hook and a test fail if anything there is ever staged or tracked, and
CI never calls paid APIs.
