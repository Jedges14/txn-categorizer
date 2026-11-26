# Prompts (E4 zero-shot, E5 few-shot)

- `system.md`: the instructions; `{categories}` is filled from `data/taxonomy.yaml` as `- Name: definition` lines.
- Each transaction is one user turn, built in `src/txncat/models/llm_fewshot.py`:
  `Description: …` / `Amount: …` / `Direction: money out | money in | unknown`.
- E5 adds k=3 examples per category (39 in total) as earlier user/assistant turns, drawn **from the train split only**
  with a fixed seed (42) and shuffled; their ids are recorded in `reports/metrics/llm_*.json`.

Rules (CLAUDE.md): the wording was written once from the taxonomy definitions and was not adjusted after looking at
val, test or external results. Few-shot examples never come from val/test/external rows.
