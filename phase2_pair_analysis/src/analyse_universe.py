"""
Runs cointegration analysis across the full instrument universe:

  1. Pairwise Engle-Granger within each economic group (static test)
  2. Bonferroni-corrected significance threshold, since we're testing
     multiple pairs at once - screen 20 pairs at the standard 0.05 level
     and you'd expect ~1 false positive by chance alone even if NOTHING
     is really related. This is a real methodological trap in stat-arb
     pair screening, not just statistical box-ticking.
  3. Johansen on each full group (N-asset test)
  4. Rolling Engle-Granger on the most promising pair, to see stability

Run: python -m src.analyze_universe
(requires data/clean/adj_close_wide.parquet - run fetch.py and clean.py first)
"""

import itertools

import pandas as pd

from src.config import UNIVERSE
from src.loader import get_prices
from src.cointegration import engle_granger_test, johansen_test
from src.rolling_analysis import rolling_engle_granger, summarize_stability


def scan_pairs_in_group(prices: pd.DataFrame, group_name: str, tickers: list[str]) -> pd.DataFrame:
    pairs = list(itertools.combinations(tickers, 2))
    bonferroni_alpha = 0.05 / len(pairs) if pairs else 0.05

    rows = []
    for a, b in pairs:
        result = engle_granger_test(prices[a], prices[b])
        rows.append({
            "group": group_name,
            "pair": f"{a}-{b}",
            "beta": round(result.beta, 3),
            "pvalue": round(result.coint_pvalue, 4),
            "cointegrated_naive_0.05": result.coint_pvalue < 0.05,
            "cointegrated_bonferroni": result.coint_pvalue < bonferroni_alpha,
        })

    return pd.DataFrame(rows)


def run():
    print("Loading clean price data...")
    prices = get_prices()
    print(f"  {len(prices)} rows, {prices.shape[1]} tickers, "
          f"{prices.index.min().date()} -> {prices.index.max().date()}\n")

    all_pair_results = []

    for group_name, tickers in UNIVERSE.items():
        available = [t for t in tickers if t in prices.columns]
        if len(available) < 2:
            print(f"[skip] {group_name}: not enough tickers with data")
            continue

        print("=" * 70)
        print(f"GROUP: {group_name}  ({', '.join(available)})")
        print("=" * 70)

        # --- pairwise Engle-Granger ---
        pair_df = scan_pairs_in_group(prices, group_name, available)
        print(pair_df.to_string(index=False))
        all_pair_results.append(pair_df)

        # --- Johansen on the full group ---
        try:
            jh = johansen_test(prices[available])
            print(f"\n{jh}")
        except Exception as e:
            print(f"\nJohansen failed: {e}")

        print()

    summary = pd.concat(all_pair_results, ignore_index=True)
    summary.to_csv("data/clean/pairwise_cointegration_summary.csv", index=False)
    print(f"Saved full pairwise summary -> data/clean/pairwise_cointegration_summary.csv\n")

    # --- rolling analysis on the most significant surviving pair ---
    survivors = summary[summary["cointegrated_bonferroni"]].sort_values("pvalue")
    if survivors.empty:
        print("No pairs survived Bonferroni correction - nothing to roll forward.")
        return

    best = survivors.iloc[0]
    a, b = best["pair"].split("-")
    print("=" * 70)
    print(f"ROLLING ANALYSIS on strongest surviving pair: {a}-{b}")
    print("=" * 70)

    rolling = rolling_engle_granger(prices[a], prices[b])
    stability = summarize_stability(rolling)
    print(f"Time cointegrated: {stability['pct_time_cointegrated']}%")
    print(f"Beta std dev: {stability['beta_std']}  (range: {stability['beta_range']})")
    print(f"Longest consecutive breakdown: {stability['longest_breakdown_windows']} windows")

    rolling.to_csv(f"data/clean/rolling_{a}_{b}.csv")
    print(f"\nSaved rolling series -> data/clean/rolling_{a}_{b}.csv "
          f"(plot pvalue and beta over time next)")


if __name__ == "__main__":
    run()