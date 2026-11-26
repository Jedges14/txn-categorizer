-- External test set support (M1). Idempotent; never drops data.
-- External rows come from a dataset with no amounts or dates, so those become nullable for that split only,
-- and the debit/credit direction it does provide is kept in its own column.
ALTER TABLE transactions ALTER COLUMN posted_at DROP NOT NULL;
ALTER TABLE transactions ALTER COLUMN amount DROP NOT NULL;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS direction TEXT;

ALTER TABLE transactions DROP CONSTRAINT IF EXISTS transactions_direction_check;
ALTER TABLE transactions ADD CONSTRAINT transactions_direction_check
  CHECK (direction IN ('debit', 'credit'));

ALTER TABLE transactions DROP CONSTRAINT IF EXISTS transactions_split_check;
ALTER TABLE transactions ADD CONSTRAINT transactions_split_check
  CHECK (split IN ('train', 'val', 'test', 'real_test', 'external_test'));

-- Everything except the external test set must still have a date and an amount.
ALTER TABLE transactions DROP CONSTRAINT IF EXISTS transactions_required_fields_check;
ALTER TABLE transactions ADD CONSTRAINT transactions_required_fields_check
  CHECK (split = 'external_test' OR (posted_at IS NOT NULL AND amount IS NOT NULL));
