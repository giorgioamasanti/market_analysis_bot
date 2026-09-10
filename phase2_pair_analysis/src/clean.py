"""
Turns the per-ticker raw parquet files into ONE wide, aligned table:
one row per trading day, one column per ticker, adjusted close prices.

This alignment step matters more than it looks. Cointegration tests,
the Kalman filter, and the HMM all assume your two (or N) series are
sampled at the exact same timestamps. If XLE has a data point on a day
GLD doesn't (holiday mismatches between exchanges, a missing print,
etc), a naive concat will silently misalign the rows and every
downstream stat will be quietly wrong - this is a classic, hard-to-spot
bug in retail quant projects. We fix it explicitly here:

  1. Load every raw ticker file
  2. Pivot into wide format on trading date
  3. Reindex all series onto the FULL business-day calendar spanned by
     the data
  4. Forward-fill small gaps (a single missing print), but cap it - if
     a ticker is missing for a long stretch, we leave NaN rather than
     silently carrying a stale price forward for weeks
  5. Drop the leading period before the youngest ticker in the group
     has data (can't test cointegration on rows where one side is NaN)

Run: python -m src.clean
"""

import pandas as pd

from src.config import RAW_DIR, CLEAN_DIR, all_tickers

MAX_FORWARD_FILL_DAYS = 3  # tolerate a short gap, don't paper over a real outage


def load_raw(ticker: str) -> pd.Series | None:
    path = RAW_DIR / f"{ticker}.parquet"
    if not path.exists():
        print(f"  [skip] {ticker}: no raw file, run fetch.py first")
        return None

    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # Adjusted close: accounts for splits/dividends so the price series
    # reflects true economic return, not corporate-action artifacts.
    series = df["Adj Close"].rename(ticker)
    return series


def build_wide_table(tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or all_tickers()

    series_list = []
    for t in tickers:
        s = load_raw(t)
        if s is not None:
            series_list.append(s)

    if not series_list:
        raise RuntimeError("No raw data found - run `python -m src.fetch` first.")

    wide = pd.concat(series_list, axis=1)

    # Reindex onto a full business-day calendar spanning the data, so
    # every ticker is evaluated at the same set of timestamps.
    full_range = pd.bdate_range(wide.index.min(), wide.index.max())
    wide = wide.reindex(full_range)
    wide.index.name = "Date"

    # Bounded forward-fill: fixes single missing prints without hiding
    # genuine multi-day gaps (delistings, trading halts, bad tickers).
    wide = wide.ffill(limit=MAX_FORWARD_FILL_DAYS)

    # Drop rows before every column has at least one real observation -
    # can't run a pairwise test where one side is entirely NaN.
    wide = wide.dropna(how="any")

    return wide


def report_coverage(wide: pd.DataFrame) -> None:
    print("\nCoverage summary:")
    print(f"  date range: {wide.index.min().date()} -> {wide.index.max().date()}")
    print(f"  rows: {len(wide)}")
    for col in wide.columns:
        n_nan = wide[col].isna().sum()
        pct = 100 * n_nan / len(wide)
        flag = "  <-- check this" if pct > 1 else ""
        print(f"  {col:8s} missing: {n_nan:4d} ({pct:.1f}%){flag}")


if __name__ == "__main__":
    wide = build_wide_table()
    report_coverage(wide)

    out_path = CLEAN_DIR / "adj_close_wide.parquet"
    wide.to_parquet(out_path)
    print(f"\nSaved clean table -> {out_path}")