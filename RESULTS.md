# Results

Every number here came from a command run in this repo, with the date and git commit.

**Machine:** Windows 11 laptop, Intel Core i7-13700H, Python 3.12.10, PostgreSQL 16.16 (Docker).

## M1: Data

Date: 2026-11-26 · Commit `9a8dd30` (CI green) · Commands:

```bash
uv run txncat data build    # seed 42
uv run txncat data load
uv run txncat data report
```

- **20,000 synthetic transactions** loaded into PostgreSQL (`SELECT count(*) FROM transactions` → 20000):
  train 13,729 (68.6%) · val 3,067 (15.3%) · test 3,204 (16.0%).
- **4,997 merchants** in `data/merchants.csv` (384 per category on average, 277–400), split 70/15/15 by merchant
  within each category; **0 merchants appear in more than one split** (`tests/test_data.py`, `notebooks/01_eda.ipynb`).
- **13 categories**, imbalanced: largest/smallest = **4.6×** (Meals & Entertainment 2,842 rows vs Rent & Lease 613).
  Per-category counts: [`reports/data_summary.md`](reports/data_summary.md).
- **Shortcut check** (share of a category's rows containing an obvious cue word; `notebooks/01_eda.ipynb`):
  Bank Fees 0.64 · Shipping 0.61 · Utilities 0.52 · Rent & Lease 0.57 · Travel 0.63 · Fuel & Auto 0.60 ·
  Professional Services 0.66. The synthetic set is likely easier than real statements; see
  [`data/README.md`](data/README.md#4-known-limitation-the-synthetic-set-may-be-easier-than-reality).
- **Real test set:** none (0 rows). No public, licensed dataset of real labeled business transactions was found, and
  no private one is available.
- **External test set** (added after the entry above; see the next section).
- Existing-dataset search: documented in [`data/README.md`](data/README.md#1-existing-datasets-searched-not-used-for-training);
  none used for training.

## M1: External test set

Date: 2026-11-26 · Commit `322d05e` · Commands:

```bash
uv run txncat data external   # DoDataThings/us-bank-transaction-categories-v2 @ 3e8d052c, SHA-256 0424aed6…
uv run txncat data load
uv run txncat data report
```

- **2,100 rows** loaded as split `external_test`: 300 each for Meals & Entertainment, Travel, Utilities, Rent & Lease,
  Bank Fees & Charges, Income / Deposits, Other (7 of 13 categories). 0 rows have an amount or date (none exist in the
  source); `direction` is set on all.
- **19.7%** (414/2,100) mention a merchant name from the train split: Travel 103, Income 98, Bank Fees 82,
  Utilities 72, Meals 30, Other 29, Rent 0.
- It is an independent **synthetic** dataset, not real data; see [`data/README.md`](data/README.md#6-external-test-set-txncat-data-external-then-txncat-data-load).

No models have been trained yet; M2 adds the first metrics.
