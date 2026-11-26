# Data

| File | What | In git? |
|---|---|---|
| `taxonomy.yaml` | The 13 categories, each with a definition and 3 example merchants. Fixed before labeling. | yes |
| `merchants.csv` | 4,997 merchants (curated real chains + generated local businesses), each with its category. | yes |
| `generated/synthetic.csv` | 20,000 synthetic transactions with their split and merchant. Rebuild with `txncat data build`. | no (generated) |
| `sample/real_test_example.csv` | 20 **fake** rows showing the exact format of the private real test set. | yes |
| `private/real_test.csv` | Your hand-labeled real transactions. | **never** (git-ignored, guarded by a pre-commit hook and a test) |

## 1. Existing datasets: searched, not used for training

Searched Hugging Face and Kaggle on 2026-11-26 for an English, labeled transaction-description dataset with a license
that allows redistribution, ≥5k rows and ≥8 categories (SPEC §3b).

| Dataset | License | Verdict |
|---|---|---|
| [DoDataThings/us-bank-transaction-categories-v2](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2) | MIT | Best candidate: 68k rows, realistic US bank strings. **But** it is itself synthetic, has **no amount or date** (SPEC's SQL features need both), and its 17 *personal-finance* labels (Groceries, Mortgage, Personal Care…) have no counterpart for 4 of our 12 business categories (Office Supplies, Advertising & Marketing, Professional Services, Shipping & Postage). Not used for training. Could serve later as an extra out-of-distribution check on the categories that map cleanly. |
| [mitulshah/transaction-categorization](https://huggingface.co/datasets/mitulshah/transaction-categorization) | MIT (card) | Gated; card is vague about which rows are real; descriptions are clean phrases ("Uber Ride"), not bank strings; 5 countries/currencies. Not used. |
| Kaggle results (e.g. "Financial Transaction Description Dataset", "Bank Transaction Data") | MIT / CC0 | Indian narrations, fraud labels, or unlabeled. None fit. |

## 2. Synthetic training data (`txncat data build`)

Code: [`src/txncat/data/merchants.py`](../src/txncat/data/merchants.py),
[`src/txncat/data/generate_synthetic.py`](../src/txncat/data/generate_synthetic.py). One seed (42) drives everything,
so the same command always produces the same file.

- **Merchants.** Each category has curated real-world chains as they appear on statements (e.g. `STARBUCKS`,
  `PADDLE.NET*`, `FACEBK ADS`) plus generated local businesses from word banks (`HARRIS AUTO REPAIR`,
  `TAMPA SELF STORAGE`); 384 per category on average, 277–400 (Bank Fees: 15 banks × 20 fee types + 20 generic). Names are unique across
  categories. Brand names are used only to describe how they appear on statements.
- **Formatting.** ALL CAPS; truncation to a random 22–32 characters (bank descriptor widths); processor prefixes
  (`SQ *`, `TST*` only for restaurants, `CLV*`, `PAYPAL *`, `POS DEBIT`, `CHECKCARD 0412`, `PADDLE.NET*` for SaaS);
  store numbers for chains; city/state suffixes; ACH-style strings for utilities, rent and invoices
  (`ACH DEBIT …`, `… AUTOPAY`); payout strings for income (`STRIPE TRANSFER ST-…`, `ZELLE FROM …`).
- **Amounts.** Category-specific log-normal (e.g. meals median ≈ $16, rent ≈ $1,340); fees from a fixed list; money
  out is negative, income positive; ~1.5% of expenses are refunds (`RETURN …`, positive amount, same category).
- **Dates.** 2024-01-01 to 2025-12-31. Built-in calendar patterns: rent on the 1st–3rd, subscriptions and utilities on
  a fixed day per merchant, fees on the 24th–28th, a mild Fri–Sun lean for meals.
- **Imbalance.** Category shares from 14% (meals) to 3% (rent); largest/smallest ≈ 4.6×.
- **No LLM paraphrasing** was used (SPEC lists it as optional); nothing about this data involved a paid API.

## 3. Splits: by merchant (leakage guard)

Each category's merchants are shuffled (seed 42) and divided 70/15/15 into train/val/test; **all rows of a merchant
go to one split**, so test measures generalization to merchants the model has never seen. Row shares come out at
68.6% / 15.3% / 16.0% because merchant popularity varies. `tests/test_data.py` asserts that no merchant appears in
two splits and that every split contains every category. Counts per category and split:
[`reports/data_summary.md`](../reports/data_summary.md); chart: `reports/figures/class_distribution.png`.

## 4. Known limitation: the synthetic set may be easier than reality

Measured in [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb): in several categories, more than half the
descriptions literally contain a category cue word: Bank Fees `FEE` 64%, Professional Services
`CPA|LAW|LLP|CONSULT|…` 66%, Travel `HOTEL|INN|AIR|…` 63%, Shipping `SHIP|POST|MAIL|…` 61%. Real statements contain
such words too, but probably less often. **This is why every headline number must also be reported on the real test
set**, and why a large LLM win on synthetic data will be checked against this before being believed.

## 5. The private real test set (you label this)

1. Export 300–500 of your own (or consenting friends') card/bank transactions as CSV.
2. **Remove** account numbers, card numbers, names of people and anything else identifying. Keep the merchant text.
3. Save as `data/private/real_test.csv` with exactly these columns (see `sample/real_test_example.csv`):

   ```
   posted_at,description,amount,category
   2025-03-03,SQ *BLUE BOTTLE COFFEE O,-6.25,Meals & Entertainment
   ```

   - `posted_at`: `YYYY-MM-DD`
   - `description`: the bank's text, unchanged apart from removing identifiers
   - `amount`: negative for money out, positive for money in, no `$` sign
   - `category`: exactly one of the 13 names in `taxonomy.yaml` (copy-paste them; `Other` if nothing fits)
4. Run `txncat data load`. It validates every row and lists problems by line number before loading anything.

Label from the description and amount alone, as the models will see them; when unsure, decide by the taxonomy
definitions, not by what the purchase "really" was. Rows stay on your machine: the folder is git-ignored, a
pre-commit hook blocks committing it, and nothing in this repo sends it anywhere.
