-- Transactions table (SPEC §3). Idempotent: safe to run more than once.
CREATE TABLE IF NOT EXISTS transactions (
  id            BIGSERIAL PRIMARY KEY,
  posted_at     TIMESTAMP NOT NULL,
  description   TEXT NOT NULL,            -- raw bank string
  amount        NUMERIC(12,2) NOT NULL,   -- negative = money out
  category      TEXT,                     -- label (NULL if unlabeled)
  split         TEXT CHECK (split IN ('train','val','test','real_test')),
  source        TEXT NOT NULL             -- 'synthetic' | '<dataset name>' | 'real'
);

CREATE INDEX IF NOT EXISTS transactions_split_idx ON transactions (split);
