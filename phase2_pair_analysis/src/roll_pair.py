"""
Standalone tool: run the rolling Engle-Granger analysis on any specific pair,
regardless of whether it passed the static (whole-sample) test. This matters
precisely because a static test averages over the WHOLE window - if two
assets were cointegrated for a while and then diverged, a single p-value
over the full sample can completely hide that, or make a genuinely
time-varying relationship look like "no relationship" at all.

Usage:
    python -m src.roll_pair GDX GDXJ
    python -m src.roll_pair GDX GDXJ --window 126   # ~6 month window instead of 1yr
"""

import argparse

from src.loader import get_prices
from src.rolling_analysis import rolling_engle_granger, summarize_stability
from src.null_benchmark import benchmark_against_null


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker_a")
    parser.add_argument("ticker_b")
    parser.add_argument("--window", type=int, default=252,
                         help="trailing window length in trading days (default 252, ~1yr)")
    parser.add_argument("--step", type=int, default=5,
                         help="how many days to step forward between windows (default 5)")
    args = parser.parse_args()

    prices = get_prices(tickers=[args.ticker_a, args.ticker_b])
    if args.ticker_a not in prices.columns or args.ticker_b not in prices.columns:
        raise ValueError(f"One or both tickers not found in clean data: "
                          f"{args.ticker_a}, {args.ticker_b}")

    rolling = rolling_engle_granger(prices[args.ticker_a], prices[args.ticker_b],
                                     window=args.window, step=args.step)
    stability = summarize_stability(rolling)

    print(f"Rolling Engle-Granger: {args.ticker_a} ~ {args.ticker_b} "
          f"(window={args.window}d, step={args.step}d)\n")
    print(f"Time cointegrated: {stability['pct_time_cointegrated']}%")
    print(f"Beta std dev: {stability['beta_std']}  (range: {stability['beta_range']}, "
          f"CV: {stability['beta_cv_pct']}%)")
    print(f"Longest consecutive breakdown: {stability['longest_breakdown_windows']} windows\n")

    # Automatic significance benchmark - is this % time cointegrated
    # actually distinguishable from what two UNRELATED assets would show
    # by chance, tested with this exact window/step? See null_benchmark.py.
    print("Significance check (vs simulated unrelated-asset null):")
    bench = benchmark_against_null(rolling, window=args.window, step=args.step)
    print(f"  {bench}\n")

    # Print the actual transitions - where did it flip from cointegrated to
    # not, or vice versa? This is the literal answer to "where does it break".
    print("Regime transitions:")
    prev = None
    for date, row in rolling.iterrows():
        cur = row["is_cointegrated"]
        if cur != prev:
            state = "COINTEGRATED" if cur else "not cointegrated"
            print(f"  {date.date()}  ->  {state}  (p={row['pvalue']:.4f}, beta={row['beta']:.3f})")
        prev = cur

    out_path = f"data/clean/rolling_{args.ticker_a}_{args.ticker_b}.csv"
    rolling.to_csv(out_path)
    print(f"\nFull series saved -> {out_path} (plot pvalue and beta over time from this)")


if __name__ == "__main__":
    main()