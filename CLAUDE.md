# Working rules
- Read SPEC.md first. Implement only the milestone I ask for; stop and summarize when it's done.
- Never invent results. Every metric in README.md or RESULTS.md must come from a command you actually ran in this repo, and RESULTS.md records that command, the date and the git commit.
- If a number looks too good (e.g. >99% accuracy), assume a bug or data leak and investigate before reporting it.
- Never commit secrets, API keys, .env files, personal financial data or data that the license forbids redistributing. Use .env.example with placeholder values.
- Write tests alongside code, not at the end. A milestone is not done until its tests pass locally and in CI.
- Use small, focused commits with clear messages (Conventional Commits style: feat:, fix:, test:, docs:, chore:).
- Prefer simple, well-known libraries over clever ones. Pin dependency versions.
- Check current docs for library/model/API names and versions rather than relying on memory; model names and free-tier hosting terms change.
- Ask me before: adding a paid service, changing the stack in SPEC.md, deleting data, or force-pushing.

# Project rules
- Never let test or real_test data influence features, hyperparameters, few-shot examples or prompt wording. Tune on val only.
- Few-shot examples come only from train, chosen with a fixed seed.
- `data/private/` must stay in `.gitignore`. Check this before every commit.
- No paid API calls in tests or CI; use recorded fixtures.
- If the LLM beats everything by a large margin on the synthetic set, check whether the synthetic data is too easy (e.g. merchant names that state the category) and say so in RESULTS.md.
