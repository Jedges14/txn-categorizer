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

## M2: SQL features and baselines (E0–E2)

Date: 2026-11-26 · Code: commit `9dda842` (the run was recorded as `7a0ea85+uncommitted`, i.e. this exact code before
it was committed) · Command: `uv run txncat train baseline` (641 s on the laptop above) · Raw:
[`reports/metrics/baseline.json`](reports/metrics/baseline.json), predictions in [`predictions/`](predictions/).

Hyperparameters chosen on **val only**: E1 `C=0.3` (of 0.3/1/3/10, class-balanced); E2 `max_depth=4,
learning_rate=0.1, min_child_weight=1`, 278 trees by early stopping (best of a 12-config grid), on the top 5,000
TF-IDF columns by chi-squared (selected on train) + 11 SQL features. Test sets were scored once, after selection.

| Model | val macro-F1 | test accuracy | **test macro-F1** | external accuracy | **external macro-F1** | external, unseen merchants (n=1,686) |
|---|---|---|---|---|---|---|
| E0 majority class (Meals) | 0.021 | 0.135 | 0.018 | 0.143 | 0.036 | 0.039 |
| E0 keyword rules | 0.674 | 0.662 | 0.666 | 0.529 | 0.525 | 0.510 |
| E1 logistic regression, TF-IDF | 0.852 | 0.854 | 0.839 | 0.531 | **0.576** | 0.469 |
| E2 XGBoost, TF-IDF + SQL features | 0.877 | 0.886 | **0.875** | 0.333 | 0.364 | 0.329 |

Test = 3,204 synthetic rows from merchants never seen in training. External = 2,100 rows from an independent dataset,
7 of 13 categories, no amounts or dates (macro-F1 averages those 7 classes).

**Reading it**
- On the synthetic test split, E2 beats E1 by 0.036 macro-F1. Per-class E2 test F1 ranges from 0.73 (Advertising &
  Marketing) to 1.00 (Income / Deposits). Income ≈ 1.00 and Bank Fees 0.99 were checked: Income is the only category
  with money coming in (refunds carry a REFUND/RETURN prefix), and 64% of fee rows say "FEE" (M1 shortcut check).
  These are properties of the synthetic data, not a leak, and are why the external set matters.
- **On the external set the ranking flips: E2 collapses (0.364) below the keyword rules (0.525), and E1 is best
  (0.576).** E2 predicts "Professional Services" for 1,003 of 2,100 external rows.
- All models score lower on external rows whose merchant was never seen in training (E1 0.576 → 0.469).

**Diagnosis of the E2 collapse** (post-hoc, does not replace the headline E2; same selected hyperparameters, no
re-tuning) · Command: `uv run python scripts/diagnose_e2_external.py` · Raw:
[`reports/metrics/e2_external_diagnostic.json`](reports/metrics/e2_external_diagnostic.json)

| E2 variant | test macro-F1 | external macro-F1 |
|---|---|---|
| all features (headline) | 0.875 | 0.364 |
| without string-format features (`desc_len`, `digit_ratio`, `has_processor_prefix`) | 0.873 | 0.378 |
| **without amount and date features** | 0.830 | **0.594** |
| text only | 0.830 | 0.527 |

- First hypothesis (string-format features don't transfer) was **wrong**: removing them changes little.
- Cause: **missing values never seen in training.** Every training row has an amount and a date; every external row
  has neither, so the trees send all of them down whichever branch they defaulted to. Without amount/date features,
  E2 recovers to 0.594 (above E1) at a cost of 0.045 on the synthetic test.
- Takeaway: amount and date features help (+0.045 macro-F1) when present, but a model must be trained to cope when
  they're absent. Planned fix, chosen on principle and to be validated on **val with amounts masked**, never on the
  external set: train with randomly masked amount/date features (missingness augmentation).

## M3: Fine-tuned encoder (E3)

Date: 2026-11-26 · Code: commit `7475d2d` (run recorded as `ffb46f9+uncommitted`: this code before commit) ·
Command: `uv sync --extra encoder && uv run txncat train finetune --seeds 0 1 2 --epochs 3` · Hardware: the laptop
CPU above (14 torch threads, no GPU), about 14 min per seed · Raw: [`reports/metrics/e3_finetune.json`](reports/metrics/e3_finetune.json),
predictions `predictions/E3_finetune_seed{0,1,2}.csv` · Model card: [`docs/model_card_e3.md`](docs/model_card_e3.md).

DistilBERT (uncased, revision `12040acc`), text only (`desc_clean`), class-weighted loss, lr 5e-5, batch 32,
3 epochs; best epoch per seed chosen on **val** (epochs 3, 2, 3). No other tuning.

| Seed | val macro-F1 | test macro-F1 | external macro-F1 | external, unseen merchants |
|---|---|---|---|---|
| 0 | 0.879 | 0.858 | 0.608 | 0.532 |
| 1 | 0.883 | 0.831 | 0.619 | 0.556 |
| 2 | 0.881 | 0.860 | 0.611 | 0.539 |
| **mean ± std** | 0.881 ± 0.002 | **0.850 ± 0.016** | **0.613 ± 0.005** | **0.542 ± 0.013** |

Accuracy, mean ± std: test 0.871 ± 0.021, external 0.585 ± 0.011.

**All models so far** (macro-F1; E3 is the 3-seed mean):

| Model | test | external | external, unseen merchants |
|---|---|---|---|
| E0 keyword rules | 0.666 | 0.525 | 0.510 |
| E1 logistic regression, TF-IDF | 0.839 | 0.576 | 0.469 |
| E2 XGBoost, TF-IDF + SQL features | **0.875** | 0.364 | 0.329 |
| E3 fine-tuned DistilBERT | 0.850 ± 0.016 | **0.613 ± 0.005** | **0.542 ± 0.013** |

**Reading it**
- On the synthetic test split, E2 is still highest; E3 sits between E1 and E2, and its seed-to-seed spread (0.831–0.860)
  is as large as some model-to-model gaps. Whether these differences are significant is M5's job (McNemar's test,
  bootstrap CIs), so no ranking claim is made here.
- On the independent external set, E3 is best of all models and most stable across seeds (std 0.005), with the
  largest margin on merchants never seen in training (0.542 vs E1 0.469). Being text-only, it avoids E2's
  missing-feature failure.
- Every model loses a lot moving from synthetic test to external data (E3: 0.85 → 0.61). That gap, not the synthetic
  scores, is the main limitation to report.
