# Commit hash map

On 2026-11-26 this repository's history was rewritten to remove "Co-Authored-By: Claude" trailers from commit messages. File contents, authors and dates are unchanged, but every rewritten commit got a new hash. Markdown docs were updated to the new hashes; the `commit` fields inside `reports/metrics/*.json` (and `bench/results/*.json`) still hold the hash recorded at run time. Use this table to resolve them.

| old | new | subject |
|---|---|---|
| `07148bb` | `6d0a771` | docs(results): M2 baselines, per-class check, external diagnosis |
| `08eadad` | `674c58c` | docs(results): M3 fine-tuned encoder results and comparison table |
| `0c5302a` | `83ed1b5` | build: add scikit-learn, xgboost, numpy, scipy; require Python 3.12 (numpy 2.5) |
| `0f9287d` | `a912200` | chore: scaffold project (uv, PostgreSQL 16 compose, transactions schema, txncat CLI) |
| `1982e05` | `7ccb3e1` | docs(data): no real test set exists; document the external test set and its limits |
| `2603205` | `9528a02` | feat(llm): run on GPU in bf16 when available; device and dtype recorded and part of the cache key |
| `2db2c33` | `a85ee64` | ci: GitHub Actions (ruff, mypy, pytest with Postgres service) and pre-commit hooks |
| `301e5f0` | `63d2e83` | feat(results): E0-E2 predictions and metrics from 9dda842 |
| `322d05e` | `9056c3b` | feat(data): external test set from an independent MIT dataset |
| `35bc18d` | `97828c7` | docs: CI badge and M0 status |
| `3f6b858` | `411bc18` | test: offline E3 smoke test (tiny model, 1 step, CPU) and seed determinism |
| `54f8834` | `332a8ff` | Initial commit |
| `56529c2` | `6eb4259` | test: CLI, schema (real Postgres) and repo-hygiene tests |
| `5798734` | `df15750` | feat(llm): E4/E5 with a free local open-weight LLM (Qwen2.5-1.5B-Instruct, Apache-2.0) |
| `7475d2d` | `95b7df3` | feat(models): E3 fine-tuned DistilBERT; txncat train finetune and data export-encoder |
| `7a0ea85` | `46d8cf3` | docs(results): record external test set numbers from 322d05e |
| `7cacdaf` | `7ece90c` | feat(analysis): diagnose E2 external collapse (missing amount/date never seen in training) |
| `8fc8ef1` | `dbbf503` | feat(results): E3 predictions and metrics, 3 seeds, from 7475d2d |
| `9a8dd30` | `7501122` | docs(data): dataset search outcome, generator method, limitations, executed EDA notebook |
| `9dda842` | `7f1b315` | feat(models): E0 majority/rules, E1 logistic TF-IDF, E2 XGBoost; txncat train baseline |
| `a8bc8bd` | `1e5522e` | docs(results): record M1 data numbers from 9a8dd30 |
| `b0b2bec` | `2e4f16e` | feat(data): taxonomy, merchant list and synthetic generator with merchant-grouped splits |
| `b1d86f5` | `b382c14` | test(data): leakage, determinism, formatting, validation and loading tests |
| `be57212` | `c71183b` | test(data): external mapping, sampling, checksum, overlap and schema constraint tests |
| `c789a58` | `a2dbad2` | feat(features): SQL cleaning and feature views with train-only statistics; TF-IDF |
| `c904244` | `1e8f5e6` | test: 13 tricky SQL cleaning strings, train-only stats, metrics, rules, val-only selection |
| `d1e88d2` | `fc124f0` | docs: fix README badge line |
| `df1054f` | `abd21da` | docs: E3 model card notes and Colab GPU notebook (unexecuted) |
| `eb68f94` | `46a7f21` | build: optional encoder extra (CPU torch 2.15, transformers 5.24); CI installs it |
| `f6d60e5` | `13aca7a` | feat(data): Postgres loader, real-test validation, data report and CLI commands |
| `ffb46f9` | `678913d` | test: make feature-view test independent of loaded data (fixes CI on an empty database) |
