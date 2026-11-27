Title: M5: significance tests, bootstrap CIs, cost/latency, txncat report

## What

M5 from SPEC §7: statistics, cost/latency, and a `txncat report` command that regenerates every table and figure from saved predictions (no DB, model or network).

- `src/txncat/eval/significance.py`: exact McNemar test on discordant pairs, Holm correction, and seeded percentile bootstrap 95% CIs for macro-F1 plus a paired ΔF1 CI. Vectorized macro-F1, checked against sklearn in tests.
- `src/txncat/eval/report.py` (`txncat report`): `reports/results_tables.md`, `reports/metrics/significance.json`, and these figures:
  - `f1_ci.png`
  - `latency_vs_f1.png`
  - confusion matrices for E2 on test, E2 on external, and E5 on external
- `src/txncat/eval/latency.py` (`txncat bench latency`): batch-size-1 p50/p95 for E1–E3 on CPU, and for E3 on GPU too.
- 11 new tests (100 total).

This branch also carries the two M4 commits that missed PR #1: the E4/E5 results and docs.

## Results (full tables in RESULTS.md)

- **Synthetic test:**
  - E2 XGBoost (0.875) and E3 DistilBERT (0.850 ± 0.016) are statistically tied (Holm p = 0.9).
  - Both beat the LLMs (p < 1e-40).
- **Independent external set:** few-shot E5 (0.667 [0.649, 0.685]) significantly beats every trained model, including E3 on all 3 seeds (Holm p ≤ 0.0012).
- **Few-shot vs zero-shot:** significant on test; not significant by McNemar on external.
- **Latency per prediction (p50):** E1 1.4 ms, E2 4.8 ms, E3 11 ms on CPU / 5.2 ms on GPU, E5 190 ms on GPU.
- **API cost:** $0 for every model.

## Test evidence

- `pytest`: 100 passed locally (DB tests ran against Postgres)
- ruff and mypy clean
