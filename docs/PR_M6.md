Title: M6: error analysis, results notebook and final README

## What

M6 from SPEC §8.

- **`txncat errors`** (`src/txncat/eval/errors.py`, `errors_cli.py`):
  - It sorts the best external-set model's (E5, few-shot LLM) 752 errors into 6 types with deterministic rules.
  - It reports what share of each type every other model gets right, and lists a seeded sample of 30 errors.
  - It checks how often each LLM answers "Other" wrongly.
  - Outputs: `reports/error_analysis.md`, `reports/metrics/error_analysis.json`, `reports/figures/error_types.png`.
- **`notebooks/03_results.ipynb`** (executed): tables and figures, read from `reports/` only.
- **README rewrite:** a plain-English summary in the first 10 lines, then results, significance, cost/latency, error
  analysis, limitations and next steps.
- **RESULTS.md M6 section**, with the command and commit.
- 14 new tests (114 total).

Builds on `feat/m5-stats`; merge that first.

## Key finding

About half (48%) of E5's errors on the independent set answer "Other" for a recognisable merchant. Zero-shot E4 does
this for 0.3% of non-Other rows; few-shot does it for 18%, and for 88% of Square/Toast (`SQ *`, `TST*`) rows. One of
its three "Other" few-shot examples is `SQ *MEMPHIS FLORIST`. This is a correlation, so it is listed as a next step (a
selection rule validated on val) and not fixed using test data.

Fine-tuned DistilBERT is the best fixer for bank wording, refunds and category confusions (54–58% of those rows).

## Test evidence

- `pytest`: 114 passed locally (DB tests ran against Postgres)
- ruff and mypy clean
- the notebook executes with no errors
