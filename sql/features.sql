-- Feature views (SPEC §6). Re-runnable: functions/views are CREATE OR REPLACE.
--
-- Cleaning rules come from the training generator's formats and widely known card-processor codes
-- (Square SQ*, Toast TST*, Clover CLV*, PayPal PAYPAL*/PYPL*/PP*, Shopify SP, Paddle, FastSpring, 2Checkout).
-- They were NOT derived from the external or test data (CLAUDE.md: test data must not shape features).
-- Statistics that need fitting (amount deciles, merchant frequency) use split = 'train' rows only.

-- Card-processor / point-of-sale prefixes: noise in front of the merchant name.
CREATE OR REPLACE FUNCTION txncat_processor_prefix_re() RETURNS text LANGUAGE sql IMMUTABLE AS $$
  SELECT '^(SQ ?\*|TST\* ?|CLV\*|PAYPAL ?\*|PYPL ?\*|PP\*|SP |POS DEBIT |DEBIT PURCHASE |DEBIT CARD PURCHASE '
      || '|CHECKCARD \d{4} |PURCHASE \d{4} |PADDLE\.NET\* ?|FS \*|2CO\.COM\*)'
$$;

-- Payment-channel words that precede a counterparty name. Kept in desc_clean (they carry meaning, e.g. money in),
-- skipped only when picking the merchant token.
CREATE OR REPLACE FUNCTION txncat_channel_prefix_re() RETURNS text LANGUAGE sql IMMUTABLE AS $$
  SELECT '^(ACH DEBIT|ACH CREDIT|ONLINE PMT|PREAUTHORIZED DEBIT|DEPOSIT|WIRE IN|ZELLE FROM) '
$$;

-- desc_clean: upper-case; drop refund marker and processor prefix; drop numeric tokens (store numbers, refs) except
-- a leading one (brands like "76"); punctuation → space; drop a trailing "CITY ST"; collapse spaces.
CREATE OR REPLACE FUNCTION txncat_clean_desc(raw text) RETURNS text LANGUAGE sql IMMUTABLE AS $$
  SELECT btrim(regexp_replace(
    regexp_replace(
      regexp_replace(
        regexp_replace(
          regexp_replace(
            regexp_replace(upper(coalesce(raw, '')), '^(RETURN|REFUND|CREDIT) ', ''),
            txncat_processor_prefix_re(), ''),
          '[^A-Z0-9&'' ]', ' ', 'g'),                         -- punctuation (*, ., -, #, /) → space
        ' \d+\M', ' ', 'g'),                                     -- numeric tokens after the first token
      '( (SAN|SANTA|LOS|LAS|NEW|SALT LAKE|ST|FORT|EL))? [A-Z]+ '
      || '(AL|AK|AZ|AR|CA|CO|CT|DE|DC|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM'
      || '|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)$', ''),  -- trailing "CITY ST"
    ' +', ' ', 'g'))
$$;

CREATE OR REPLACE FUNCTION txncat_merchant_token(clean text) RETURNS text LANGUAGE sql IMMUTABLE AS $$
  SELECT nullif(btrim(split_part(s, ' ', 1) || ' ' || split_part(s, ' ', 2)), '')
  FROM (SELECT regexp_replace(coalesce(clean, ''), txncat_channel_prefix_re(), '') AS s) x
$$;

CREATE OR REPLACE VIEW txn_clean AS
SELECT
  t.id,
  t.split,
  t.source,
  t.category,
  t.description,
  t.posted_at,
  t.amount,
  t.direction,
  txncat_clean_desc(t.description) AS desc_clean,
  txncat_merchant_token(txncat_clean_desc(t.description)) AS merchant_token,
  upper(t.description) ~ ('^(RETURN |REFUND |CREDIT )?' || substr(txncat_processor_prefix_re(), 2)) AS has_processor_prefix,
  upper(t.description) ~ '^(RETURN|REFUND|CREDIT) ' AS is_refund
FROM transactions t;

-- Amount deciles from TRAIN rows only (NTILE-equivalent cut points), applied to every split.
CREATE OR REPLACE VIEW train_amount_cutpoints AS
SELECT percentile_cont(ARRAY[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
         WITHIN GROUP (ORDER BY abs(amount))::double precision[] AS cuts
FROM transactions
WHERE split = 'train' AND amount IS NOT NULL;

-- How often each merchant token occurs among TRAIN rows only.
CREATE OR REPLACE VIEW train_merchant_freq AS
SELECT merchant_token, count(*)::int AS n
FROM txn_clean
WHERE split = 'train' AND merchant_token IS NOT NULL
GROUP BY merchant_token;

CREATE OR REPLACE VIEW txn_features AS
SELECT
  c.id,
  c.split,
  c.source,
  c.category,
  c.description,
  c.desc_clean,
  c.merchant_token,
  abs(c.amount)::double precision AS amount_abs,
  CASE
    WHEN c.amount IS NOT NULL THEN c.amount < 0
    WHEN c.direction IS NOT NULL THEN c.direction = 'debit'
  END AS is_outflow,
  CASE WHEN c.amount IS NOT NULL THEN width_bucket(abs(c.amount)::double precision, cp.cuts) END AS amount_bucket,
  extract(isodow FROM c.posted_at)::int AS day_of_week,            -- 1 = Monday … 7 = Sunday
  (extract(isodow FROM c.posted_at) >= 6) AS is_weekend,
  extract(day FROM c.posted_at)::int AS day_of_month,
  coalesce(mf.n, 0) AS merchant_freq,
  c.has_processor_prefix,
  c.is_refund,
  length(c.description) AS desc_len,
  length(regexp_replace(c.description, '[^0-9]', '', 'g'))::double precision
    / greatest(length(c.description), 1) AS digit_ratio
FROM txn_clean c
CROSS JOIN train_amount_cutpoints cp
LEFT JOIN train_merchant_freq mf ON mf.merchant_token = c.merchant_token;
