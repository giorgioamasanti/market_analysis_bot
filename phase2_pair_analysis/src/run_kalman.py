"""
Runs the Kalman filter hedge-ratio estimation (src/kalman.py) on a real
pair, saves the full time series, and produces a diagnostic plot.

Usage:
    python -m src.run_kalman GDX GDXJ
    python -m src.run_kalman GDX GDXJ --delta 1e-5   # smoother, slower-adapting beta
    python -m src.run_kalman GDX GDXJ --delta 1e-3   # more reactive, noisier beta
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.loader import get_prices
from src.kalman import kalman_hedge_ratio
from src.rolling_analysis import rolling_engle_granger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker_a")
    parser.add_argument("ticker_b")
    parser.add_argument("--delta", type=float, default=1e-4,
                         help="how fast beta is allowed to drift (default 1e-4). "
                              "Smaller = smoother/slower, larger = more reactive/noisier.")
    parser.add_argument("--burn_in", type=int, default=30)
    parser.add_argument("--zscore_threshold", type=float, default=2.0,
                         help="threshold for flagging the current spread as extended")
    args = parser.parse_args()

    prices = get_prices(tickers=[args.ticker_a, args.ticker_b])
    if args.ticker_a not in prices.columns or args.ticker_b not in prices.columns:
        raise ValueError(f"One or both tickers not found in clean data: "
                          f"{args.ticker_a}, {args.ticker_b}")

    result = kalman_hedge_ratio(prices[args.ticker_a], prices[args.ticker_b],
                                 delta=args.delta, burn_in=args.burn_in)

    print(f"Kalman filter: {args.ticker_a} ~ {args.ticker_b} (delta={args.delta})\n")
    print(f"Beta path: {result.beta.iloc[0]:.4f} (start) -> {result.beta.iloc[-1]:.4f} (latest)")
    print(f"Beta mean/std over full history: {result.beta.mean():.4f} +/- {result.beta.std():.4f}")
    print(f"Current spread z-score: {result.zscore.iloc[-1]:.2f}")

    if abs(result.zscore.iloc[-1]) > args.zscore_threshold:
        direction = "ABOVE" if result.zscore.iloc[-1] > 0 else "BELOW"
        print(f"  -> currently {direction} the +/-{args.zscore_threshold} threshold "
              f"({args.ticker_a} looks {'rich' if direction == 'ABOVE' else 'cheap'} "
              f"relative to {args.ticker_b} given the current hedge ratio)")
    else:
        print(f"  -> within +/-{args.zscore_threshold}, no extended dislocation right now")

    # save the full series
    out_path = f"data/clean/kalman_{args.ticker_a}_{args.ticker_b}.csv"
    out_df = pd.concat([result.alpha, result.beta, result.spread,
                         result.spread_std, result.zscore], axis=1)
    out_df.to_csv(out_path)
    print(f"\nFull series saved -> {out_path}")

    # cross-check against the rolling (windowed) beta as a sanity comparison -
    # the two methods should broadly agree on DIRECTION of drift even though
    # the Kalman path is smoother/more continuous
    rolling = rolling_engle_granger(prices[args.ticker_a], prices[args.ticker_b])

    fig, (ax_beta, ax_z) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax_beta.plot(result.beta.index, result.beta.values, color="#ff7f0e", lw=1,
                 label=f"Kalman beta (delta={args.delta})")
    ax_beta.plot(rolling.index, rolling["beta"], color="#1f77b4", lw=1, ls="--",
                 alpha=0.6, label="rolling OLS beta (252d window)")
    ax_beta.set_ylabel("hedge ratio (beta)")
    ax_beta.set_title(f"Kalman filter: {args.ticker_a} ~ {args.ticker_b}")
    ax_beta.legend(loc="best", fontsize=8)

    ax_z.plot(result.zscore.index, result.zscore.values, color="#2ca02c", lw=1)
    ax_z.axhline(args.zscore_threshold, color="gray", ls="--", lw=1)
    ax_z.axhline(-args.zscore_threshold, color="gray", ls="--", lw=1)
    ax_z.axhline(0, color="gray", lw=0.5)
    ax_z.set_ylabel("spread z-score")
    ax_z.set_xlabel("date")

    fig.tight_layout()
    plot_path = f"data/clean/kalman_{args.ticker_a}_{args.ticker_b}.png"
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"Diagnostic plot saved -> {plot_path} "
          f"(Kalman beta vs rolling OLS beta, and the z-score used for signals)")


if __name__ == "__main__":
    main()