# Model card notes: E3 fine-tuned DistilBERT

**What it is.** [`distilbert/distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased)
(Apache-2.0, revision `12040acc`) with a 13-way classification head, fine-tuned to sort bank-transaction description
strings into the small-business categories in [`data/taxonomy.yaml`](../data/taxonomy.yaml). Code:
[`src/txncat/models/finetune.py`](../src/txncat/models/finetune.py). Research artifact; not for production use.

**Input.** `desc_clean` only (the cleaned description; see `sql/features.sql`), lower-cased by the tokenizer,
max 32 tokens. No amount, date or direction: E3 is a text-only model.

**Training.** Synthetic train split (13,729 rows, see [`data/README.md`](../data/README.md)). AdamW, learning rate
5e-5, weight decay 0.01, batch 32, 3 epochs, linear schedule with 10% warmup, gradient clipping at 1.0,
class-weighted cross-entropy (inverse class frequency). The epoch with the best **val** macro-F1 is kept
(seed 0 → epoch 3, seed 1 → epoch 2, seed 2 → epoch 3). No other hyperparameters were tuned. Trained on a laptop
CPU (Intel i7-13700H, 14 torch threads), about 14 minutes per seed. A Colab GPU notebook exists
(`notebooks/02_finetune_colab.ipynb`) but the reported numbers come from the CPU run.

**Evaluation.** 3 seeds (0, 1, 2); mean ± std in [`RESULTS.md`](../RESULTS.md#m3-fine-tuned-encoder-e3).
Synthetic test: macro-F1 0.850 ± 0.016. External test (independent dataset, 7 of 13 categories): 0.613 ± 0.005.

**Limitations.**
- Trained only on synthetic data from this repo's generator; there is no real-transaction evaluation.
- Seed-to-seed spread on the synthetic test is noticeable (0.831–0.860 macro-F1); compare models with the
  significance tests in M5, not single runs.
- Text-only: it cannot use amount or timing signals that help on the synthetic data (E2 shows +0.045 from them).
- External performance is far below synthetic performance (0.61 vs 0.85), and lower again on merchants never seen in
  training (0.54): it generalizes better than the classical models, but not well.
- US-centric English strings; categories reflect a small-business bookkeeping view.
