"""
Pulls daily OHLCV for every ticker in the universe and caches it as raw
parquet, one file per ticker, under data/raw/.

Design choices worth knowing:
- We cache RAW downloads untouched. Cleaning/alignment happens later in
  clean.py, on a separate copy. Never overwrite raw data with a cleaned
  version - you'll want to re-clean with different logic later without
  re-hitting the API.
- yfinance sometimes returns MultiIndex columns when downloading a single
  ticker (depends on version) - we flatten that defensively.
- We keep 'auto_adjust=False' and store separate Close vs Adj Close.
  For pairs/spread trading you generally want *adjusted* prices (splits
  and dividends distort a raw price series and will show up as fake
  "spread breaks" that are really just corporate actions, not economics).
  We'll use Adj Close downstream.

Run: python -m src.fetch
"""

import time
import pandas as pd
import yfinance as yf

from src.config import UNIVERSE, START_DATE, END_DATE, RAW_DIR, all_tickers


def fetch_ticker(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """Download one ticker's daily OHLCV. Returns empty df on failure
    rather than raising, so one bad ticker doesn't kill the whole batch."""
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            print(f"  [warn] {ticker}: no data returned")
            return pd.DataFrame()

        # yfinance can return MultiIndex columns like ('Close', 'XLE') even
        # for a single ticker depending on version - flatten defensively.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()  # 'Date' becomes a normal column
        df["ticker"] = ticker
        return df

    except Exception as e:
        print(f"  [error] {ticker}: {e}")
        return pd.DataFrame()


def fetch_all(tickers: list[str] | None = None) -> None:
    tickers = tickers or all_tickers()
    print(f"Fetching {len(tickers)} tickers from {START_DATE} to {END_DATE or 'today'}")

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}")
        df = fetch_ticker(ticker, START_DATE, END_DATE)

        if df.empty:
            continue

        out_path = RAW_DIR / f"{ticker}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  saved {len(df)} rows -> {out_path.relative_to(RAW_DIR.parents[1])}")

        time.sleep(0.5)  # be polite to the API, avoid rate limiting on a big universe


if __name__ == "__main__":
    fetch_all()