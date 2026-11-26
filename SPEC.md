# SPEC: Transaction Categorizer, Classical ML vs. LLM

Resume version: **AI Science Intern** · Repo name suggestion: `txn-categorizer`

## 1. Goal

Build an end-to-end ML project that sorts short bank-transaction descriptions (e.g. `SQ *BLUE BOTTLE 0423 OAKLAND CA`) into small-business expense categories, and rigorously compare:

- **A. Classical baseline:** SQL-engineered features + TF-IDF text features → XGBoost (and logistic regression as a sanity baseline)
- **B. Fine-tuned small transformer:** a compact encoder (e.g. DistilBERT or ModernBERT-base) fine-tuned for classification
- **C. LLM few-shot:** a hosted LLM prompted with a category list and k examples, returning structured JSON

Report accuracy, macro-F1, statistical significance, cost per 1,000 predictions and latency. Close with an error analysis.

This mirrors a real small-business bookkeeping problem, and it covers the posting's requirements: Python, SQL, data wrangling/ETL, feature engineering, classification, experiments with statistical analysis, LLMs, and clear documentation.

### Resume bullets this project must support (fill with measured numbers)

- Loaded [N] bank transactions into PostgreSQL and engineered merchant, amount and timing features in SQL.
- Trained an XGBoost baseline ([x]% macro-F1) and compared it with a [few-shot / fine-tuned] model ([x]% macro-F1); the difference was significant (p < [x], McNemar's test).
- Compared cost per 1,000 predictions and latency; error analysis showed [finding].

## 2. Non-goals

- No web app or UI beyond an optional small Streamlit demo (milestone 7).
- No real-time serving infrastructure.
- No personal or real customer financial data committed to the repo, ever.

## 3. Data plan

There is no well-known public, labeled, English bank-transaction categorization benchmark, so the dataset is built in two parts.

### 3a. Category taxonomy (fixed before any labeling)

Use ~12 small-business expense categories, loosely modeled on common bookkeeping categories:
`Meals & Entertainment, Travel, Fuel & Auto, Office Supplies, Software & Subscriptions, Utilities, Rent & Lease, Advertising & Marketing, Professional Services, Shipping & Postage, Bank Fees & Charges, Income / Deposits`, plus `Other`.
Store it in `data/taxonomy.yaml` with a one-line definition and 3 example merchants per category.

### 3b. Training data: synthetic, realistic, documented

- First, Claude Code searches Hugging Face / Kaggle for an existing labeled transaction-description dataset with a license that allows redistribution and English descriptions. If one is suitable (≥5k rows, ≥8 categories), use it as the training base and map it to the taxonomy. Document the source and license in `data/README.md`.
- Otherwise (or in addition), generate synthetic transactions with a script (`src/data/generate_synthetic.py`):
  - A curated list of ~400 real-world merchant name patterns per category (chains and generic local-business patterns), written into `data/merchants.csv`.
  - Templates that mimic real bank formatting: processor prefixes (`SQ *`, `TST*`, `PAYPAL *`, `POS DEBIT`), store numbers, truncation to 22–32 characters, city/state suffixes, ALL CAPS, random dates and amount distributions per category (log-normal, category-specific parameters).
  - Optionally, an LLM paraphrase step to add variety. Keep the seed fixed and log the prompts.
  - Target: ~20k rows, class-imbalanced on purpose to look realistic.
- **Leakage guard:** split train/val/test **by merchant**, so no merchant pattern appears in more than one split.

### 3c. Real-world test set: hand-labeled, private

- Jedges exports ~300–500 of his own (or consenting friends') real card/bank transactions, removes account numbers and names, and hand-labels them with the taxonomy.
- Stored at `data/private/real_test.csv`, **git-ignored**. The repo includes a 20-row fake example with the same schema so others can run the code.
- Every headline number is reported on **both** the synthetic test split and the real test set. The real set is the honest one. If results differ a lot, that's a finding for the write-up, not something to hide.

### Schema (PostgreSQL)

```sql
CREATE TABLE transactions (
  id            BIGSERIAL PRIMARY KEY,
  posted_at     TIMESTAMP NOT NULL,
  description   TEXT NOT NULL,          -- raw bank string
  amount        NUMERIC(12,2) NOT NULL, -- negative = money out
  category      TEXT,                   -- label (NULL if unlabeled)
  split         TEXT CHECK (split IN ('train','val','test','real_test')),
  source        TEXT NOT NULL           -- 'synthetic' | '<dataset name>' | 'real'
);
```

## 4. Tech stack

- Python 3.11+, managed with `uv` (or `pip` + `requirements.txt` if uv causes friction)
- PostgreSQL 16 in Docker (`docker-compose.yml`), accessed with SQLAlchemy + psycopg
- pandas, scikit-learn, XGBoost
- Hugging Face `transformers` + `datasets` for fine-tuning (run on CPU for smoke tests and Google Colab GPU for the real run; include a Colab notebook)
- LLM: a provider-agnostic `LLMClient` interface with one implementation for a hosted model API (default: a small, cheap model from Anthropic or OpenAI; check current model names). Structured JSON output, retries with backoff, and a disk cache keyed by (model, prompt hash) so reruns cost $0.
- statsmodels / scipy for statistical tests
- MLflow (local file store) for experiment tracking, or a simple `runs/` JSON log if MLflow is too heavy
- pytest, ruff, mypy (lenient), pre-commit
- GitHub Actions: lint + unit tests on every push (no GPU, no paid API calls in CI; LLM tests use recorded fixtures)

## 5. Repo layout

```
txn-categorizer/
  README.md  SPEC.md  CLAUDE.md  RESULTS.md
  docker-compose.yml  pyproject.toml  .env.example  .github/workflows/ci.yml
  data/README.md  data/taxonomy.yaml  data/merchants.csv  data/sample/  data/private/ (git-ignored)
  sql/
    001_schema.sql
    features.sql          -- feature views (see §6)
  src/txncat/
    data/generate_synthetic.py  load_to_postgres.py  splits.py
    features/sql_features.py  text_features.py
    models/baseline.py  finetune.py  llm_fewshot.py  llm_client.py
    eval/metrics.py  significance.py  cost_latency.py  error_analysis.py
    cli.py                -- `txncat <command>` entry point
  notebooks/
    01_eda.ipynb  02_finetune_colab.ipynb  03_results.ipynb
  tests/
  reports/figures/
```

## 6. Feature engineering (SQL-first, on purpose)

Create views in `sql/features.sql` so the SQL work is visible:

- `desc_clean`: uppercase, strip processor prefixes (`SQ *`, `TST*`, …), digits-only tokens, trailing city/state (regex in SQL)
- `merchant_token`: first 1–2 meaningful tokens after cleaning
- `amount_abs`, `is_outflow`, `amount_bucket` (NTILE over training data only)
- `day_of_week`, `is_weekend`, `day_of_month`
- `merchant_freq`: how often the merchant token appears in **training** rows (computed only from the train split to avoid leakage)
- `has_processor_prefix`, `desc_len`, `digit_ratio`

Python adds TF-IDF on `desc_clean` (word 1–2 grams + char 3–5 grams) and joins it with the SQL features.

## 7. Experiments

| ID | Model | Notes |
|----|-------|-------|
| E0 | Majority class + keyword rules | Floor baseline |
| E1 | Logistic regression, TF-IDF only | |
| E2 | XGBoost, TF-IDF + SQL features | Tune with val split only (small grid or Optuna, ≤50 trials) |
| E3 | Fine-tuned encoder | 3 seeds; report mean ± std |
| E4 | LLM zero-shot | Category list + definitions only |
| E5 | LLM few-shot (k = 3 per class, from train) | Fixed example selection, logged |
| E6 (stretch) | Hybrid: XGBoost, and fall back to LLM when confidence < threshold | Shows the cost/accuracy trade-off |

### Evaluation (in `eval/`)

- Accuracy, macro-F1, per-class F1, confusion matrix, on both `test` and `real_test`.
- **Significance:** McNemar's test for paired comparisons on the same test rows (E2 vs E3, E2 vs E5, E3 vs E5), plus 1,000-sample bootstrap 95% CIs for macro-F1.
- **Cost:** dollars per 1,000 predictions (LLM token usage × current pricing, stored in a config file with the date checked), and training compute time for E2/E3.
- **Latency:** p50/p95 per-prediction latency, measured on the same machine; note the hardware in RESULTS.md.
- **Error analysis:** the top 30 misclassified real-test rows for the best model, grouped into 4–6 error types (e.g. ambiguous merchant, processor-only string, refunds). Show which model fixes which error type.
- One figure each: macro-F1 bar chart with CIs, cost vs. macro-F1 scatter, confusion matrix for the best model.

## 8. Milestones and acceptance criteria

**M0, Setup (½ day).** Repo, uv/pyproject, docker-compose Postgres, pre-commit, CI with a passing placeholder test, CLAUDE.md, `.env.example`.
✅ `docker compose up -d && uv run pytest` passes; CI green.

**M1, Data (2–3 days).** Taxonomy, dataset search (document the result), synthetic generator, merchant-grouped splits, loader into Postgres, fake sample data, EDA notebook.
✅ `txncat data build && txncat data load` creates ≥15k rows; tests prove no merchant appears in two splits; class distribution chart in `reports/figures/`.
👤 Jedges: start hand-labeling `real_test.csv` in parallel.

**M2, SQL features + baselines (2 days).** `features.sql`, E0–E2, metrics module, first RESULTS.md entries.
✅ `txncat train baseline` prints test and real_test metrics; unit tests cover the SQL cleaning on 10 tricky strings.

**M3, Fine-tuned encoder (2 days).** Training script + Colab notebook, 3 seeds, model card notes.
✅ E3 results with mean ± std in RESULTS.md; CPU smoke test (tiny subset, 1 step) runs in CI.

**M4, LLM evaluation (2 days).** LLMClient, prompt templates in `prompts/`, JSON validation, caching, E4–E5. Set a hard spending cap in config (e.g. stop at $5).
✅ E4–E5 results; a cached rerun costs $0; invalid JSON responses are counted and reported, not silently dropped.

**M5, Statistics, cost, latency (1–2 days).** McNemar tests, bootstrap CIs, cost and latency tables, figures.
✅ `txncat report` regenerates every table and figure from saved predictions.

**M6, Error analysis + write-up (2 days).** Error taxonomy, `notebooks/03_results.ipynb`, README with results, figures, limitations (synthetic training data, small real test set) and next steps.
✅ README explains the result in plain English in its first 10 lines.

**M7, Stretch (optional).** E6 hybrid router; a tiny Streamlit demo where you paste a transaction string and see each model's prediction; a short blog post or LinkedIn write-up.

## 9. Rules for Claude Code (project-specific; add to CLAUDE.md)

- Never let test or real_test data influence features, hyperparameters, few-shot examples or prompt wording. Tune on val only.
- Few-shot examples come only from train, chosen with a fixed seed.
- `data/private/` must stay in `.gitignore`. Check this before every commit.
- No paid API calls in tests or CI; use recorded fixtures.
- If the LLM beats everything by a large margin on the synthetic set, check whether the synthetic data is too easy (e.g. merchant names that state the category) and say so in RESULTS.md.

## 10. Kickoff prompt

> Read SPEC.md and CLAUDE.md. We're building the Transaction Categorizer. Start with milestone M0 only: set up the repo structure, pyproject with uv, docker-compose for Postgres 16, pre-commit (ruff), a GitHub Actions workflow running ruff + pytest, `.env.example`, and a `.gitignore` that excludes `data/private/` and `.env`. Then stop, show me how to run it, and list what M1 will need from me.
