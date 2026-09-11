"""
Standalone tool: test cointegration between two or more assets over a
SPECIFIC, manually chosen date range - as opposed to rolling automatically.

This complements roll_pair.py rather than replacing it:
  - roll_pair.py sweeps a moving window automatically and shows you WHERE
    a transition happened.
  - region_test.py lets you zoom into a specific region you already have
    a hypothesis about (e.g. "was this pair cointegrated just during the
    2022 gold rally?" or "test only the post-2023 period") and get a full,
    precise static test result for exactly that slice - useful once
    roll_pair.py has pointed you at an interesting window and you want the
    full detail (hedge ratio, p-value, residual series) for that specific
    region rather than reading it off a rolling summary.

Usage:
    python -m src.region_test GLD GDX --start 2020-01-01 --end 2021-06-01
    python -m src.region_test GLD GDX GDXJ --start 2022-01-01 --end 2023-01-01   # Johansen (3 assets)
"""

import argparse

from src.loader import get_prices
from src.cointegration import engle_granger_test, johansen_test, adf_pvalue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="+", help="2 tickers for Engle-Granger, "
                                                     "3+ for Johansen")
    parser.add_argument("--start", required=True, help="e.g. 2020-01-01")
    parser.add_argument("--end", required=True, help="e.g. 2021-06-01")
    args = parser.parse_args()

    prices = get_prices(tickers=args.tickers, start=args.start, end=args.end)
    missing = [t for t in args.tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"Not found in clean data: {missing}")

    n = len(prices)
    print(f"Testing {', '.join(args.tickers)} over {args.start} -> {args.end} "
          f"({n} trading days)\n")

    if n < 60:
        print(f"WARNING: only {n} observations in this window - cointegration "
              f"tests need reasonable sample size to have statistical power. "
              f"Treat any result here as indicative, not conclusive.\n")

    # individual ADF checks - confirms each series is I(1) within this
    # specific sub-window (this CAN change between regions - a series that's
    # a random walk over 5 years might look locally more/less trending
    # within a shorter slice, which is itself useful context)
    print("Individual series (should be non-stationary, p > 0.05):")
    for t in args.tickers:
        p = adf_pvalue(prices[t])
        flag = "" if p > 0.05 else "  <-- looks stationary in this window, unusual"
        print(f"  {t}: ADF p={p:.4f}{flag}")
    print()

    if len(args.tickers) == 2:
        a, b = args.tickers
        result = engle_granger_test(prices[a], prices[b])
        print(result)
        print(f"  hedge ratio (beta): {result.beta:.4f}")
        print(f"  intercept (alpha): {result.alpha:.4f}")
        print(f"  spread mean: {result.residual.mean():.4f}, "
              f"std: {result.residual.std():.4f}")
    else:
        result = johansen_test(prices[args.tickers])
        print(result)
        if not result.stable:
            print(f"  (Johansen unstable in this window - fallback pair result above)")


if __name__ == "__main__":
    main()