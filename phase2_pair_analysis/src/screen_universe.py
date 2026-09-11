"""
Screens the ENTIRE universe (src/config.UNIVERSE) for genuinely promising
cointegrated pairs, cheaply, before spending time deep-diving on any one
pair with roll_pair.py / run_kalman.py.

Two-stage philosophy: this is the cheap, BROAD first pass across every
pair in the universe. Only pairs that look genuinely promising here are
worth the more expensive per-pair investigation (rolling regime plots,
Kalman filter, region tests). This is what actually answers "which pairs
are worth studying" with evidence, instead of picking one because it came
up in conversation.

Run:
    python -m src.screen_universe
    python -m src.screen_universe --step 10   # coarser/faster rolling step for the screen
"""

import argparse
import itertools
import time

import pandas as pd

from src.config import UNIVERSE
from src.loader import get_prices
from src.cointegration import engle_granger_test
from src.rolling_analysis import rolling_engle_granger, summarize_stability
from src.null_benchmark import benchmark_against_null


def screen_pair(prices: pd.DataFrame, group: str, a: str, b: str,
                 window: int, step: int) -> dict | None:
    try:
        static = engle_granger_test(prices[a], prices[b])
    except Exception:
        return None

    rolling = rolling_engle_granger(prices[a], prices[b], window=window, step=step)
    if len(rolling) < 10:
        return None  # not enough shared history for a meaningful screen

    stability = summarize_stability(rolling)
    bench = benchmark_against_null(rolling, window=window, step=step, n_trials=10)

    return {
        "group": group,
        "pair": f"{a}-{b}",
        "static_pvalue": round(static.coint_pvalue, 4),
        "pct_cointegrated": stability["pct_time_cointegrated"],
        "z_vs_null": bench.z_score,
        "null_p95": round(bench.null_p95, 1),
        "beta_cv_pct": stability["beta_cv_pct"],
        "longest_breakdown_windows": stability["longest_breakdown_windows"],
        "n_windows": len(rolling),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--step", type=int, default=10,
                         help="coarser default than roll_pair.py's step=5 - this is a "
                              "broad screen across many pairs, not a single deep dive")
    args = parser.parse_args()

    tickers = sorted(set(t for grp in UNIVERSE.values() for t in grp))
    prices = get_prices(tickers=tickers)
    print(f"Loaded {prices.shape[1]} tickers, {len(prices)} rows\n")

    pairs_to_run = [(group, a, b) for group, grp in UNIVERSE.items()
                     for a, b in itertools.combinations(
                         [t for t in grp if t in prices.columns], 2)]

    results = []
    start = time.time()
    for i, (group, a, b) in enumerate(pairs_to_run, 1):
        print(f"[{i}/{len(pairs_to_run)}] {group}: {a}-{b}...", end=" ", flush=True)
        r = screen_pair(prices, group, a, b, args.window, args.step)
        if r:
            results.append(r)
            print(f"z_vs_null={r['z_vs_null']:.1f}, beta_cv={r['beta_cv_pct']:.1f}%")
        else:
            print("skipped (insufficient overlapping history)")

    elapsed = time.time() - start
    print(f"\nScreened {len(results)}/{len(pairs_to_run)} pairs in {elapsed:.0f}s\n")

    df = pd.DataFrame(results).sort_values("z_vs_null", ascending=False)
    out_path = "data/clean/universe_screen.csv"
    df.to_csv(out_path, index=False)

    print("=" * 105)
    print("TOP CANDIDATES (sorted by strength of evidence vs a random-walk null)")
    print("=" * 105)
    print(df.head(20).to_string(index=False))

    print(f"\nFull results (all {len(results)} pairs) saved -> {out_path}")
    print("\nHow to read the columns:")
    print("  z_vs_null           higher = more clearly a genuine relationship, not noise")
    print("                      (z > ~5 is a strong bar; see null_benchmark.py for the null simulation)")
    print("  pct_cointegrated    % of rolling windows flagged cointegrated")
    print("  beta_cv_pct         hedge ratio stability - LOWER is better (a real, fixed, tradeable ratio)")
    print("  longest_breakdown_windows   longest stretch NOT cointegrated - shorter is more consistent")
    print("\nSuggested next step: take the top ~5-10 by z_vs_null that ALSO have reasonably low")
    print("beta_cv_pct (both conditions matter - see the SPY-SMH investigation for why), then run")
    print("roll_pair.py and run_kalman.py on each for the full deep-dive (regime plots, live z-score,")
    print("spread volatility) before concluding anything is a genuine trading candidate.")


if __name__ == "__main__":
    main()