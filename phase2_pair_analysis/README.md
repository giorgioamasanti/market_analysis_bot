# Quant project — phase 2: data pipeline

## Setup

```bash
pip install yfinance pandas pyarrow duckdb
```

## Usage

Run these from the `quant_project/` root, in order:

```bash
python -m src.fetch    # downloads raw OHLCV per ticker -> data/raw/*.parquet
python -m src.clean    # aligns everything into one wide table -> data/clean/adj_close_wide.parquet
python -m src.loader   # smoke test: prints last few rows for SPY/IVV/VOO
```

## What each step does

- **`src/config.py`** — the instrument universe, grouped by expected economic
  relationship (energy, financials, gold, semis, broad market). Add new
  tickers here, nowhere else.
- **`src/fetch.py`** — pulls daily OHLCV from Yahoo Finance via `yfinance`,
  caches one untouched raw parquet per ticker in `data/raw/`. Re-running it
  re-downloads everything; safe to do periodically to extend history.
- **`src/clean.py`** — the important one. Loads every raw ticker, pivots to
  a wide table (one column per ticker), reindexes onto a shared business-day
  calendar, bounded forward-fill for single missing prints, drops any
  leading NaN period. Prints a coverage report so you can see exactly how
  much data was missing per ticker before you trust anything downstream.
- **`src/loader.py`** — DuckDB query helper over the clean table. Overkill
  for one file today; becomes useful once you're partitioning multiple
  data types (this is the same interface you'll reuse for phase 4 order
  book data).

## Sanity check

The `broad_market` group (SPY, IVV, VOO) tracks the same index via three
different providers — they should show near-perfect cointegration. If your
Engle-Granger/Johansen pipeline doesn't find that relationship, the bug is
in your pipeline, not in the markets. Always validate a new pipeline against
a case you know the answer to before trusting it on a case you don't.

## A note on adjusted vs raw close

We use `Adj Close`, not `Close`. A stock split or dividend shows up as a
price jump in raw close that has nothing to do with economics — left
uncorrected, it looks exactly like a spread "breaking down" to a
cointegration test, when really it's just an accounting artifact. Always
check which price field you're using before drawing conclusions about
regime breaks.

## Network note

This was scaffolded in a sandboxed environment without access to Yahoo
Finance's hosts — it's untested against live data. Run `python -m src.fetch`
on your own machine first and check the coverage report before moving on to
the cointegration tests.