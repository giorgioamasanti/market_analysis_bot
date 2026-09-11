"""
Saves a two-panel diagnostic plot for a rolling cointegration result:
  top:    p-value over time, with the 0.05 significance line, log-scaled
          (p-values span orders of magnitude - linear scale crushes
          everything below ~0.1 into the same pixel row)
  bottom: hedge ratio (beta) over time

Both panels shade cointegrated regimes green and breakdown regimes red,
using identify_regimes() - so you can SEE where the interesting windows
are instead of reading dates off a printed table.
"""

import matplotlib
matplotlib.use("Agg")  # no display needed - just save to file
import matplotlib.pyplot as plt

import pandas as pd

from src.rolling_analysis import identify_regimes


def plot_rolling_diagnostics(rolling_result: pd.DataFrame, pair_name: str,
                              save_path: str) -> None:
    regimes = identify_regimes(rolling_result)

    fig, (ax_p, ax_beta) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                         gridspec_kw={"height_ratios": [1, 1]})

    # shade every regime on BOTH panels, so it's obvious which stretch of
    # the beta trace corresponds to which p-value regime
    for _, r in regimes.iterrows():
        color = "#2ca02c" if r["is_cointegrated"] else "#d62728"
        for ax in (ax_p, ax_beta):
            ax.axvspan(r["start"], r["end"], color=color, alpha=0.08, lw=0)

    ax_p.plot(rolling_result.index, rolling_result["pvalue"], color="#1f77b4", lw=1)
    ax_p.axhline(0.05, color="gray", ls="--", lw=1, label="p=0.05 threshold")
    ax_p.set_yscale("log")
    ax_p.set_ylabel("p-value (log scale)")
    ax_p.set_title(f"Rolling Engle-Granger: {pair_name}")
    ax_p.legend(loc="upper right", fontsize=8)

    ax_beta.plot(rolling_result.index, rolling_result["beta"], color="#ff7f0e", lw=1)
    ax_beta.set_ylabel("hedge ratio (beta)")
    ax_beta.set_xlabel("date")

    # annotate the single longest cointegrated regime directly on the chart -
    # the most likely "look here" candidate
    coint_regimes = regimes[regimes["is_cointegrated"]]
    if not coint_regimes.empty:
        longest = coint_regimes.loc[coint_regimes["n_windows"].idxmax()]
        mid = longest["start"] + (longest["end"] - longest["start"]) / 2
        ax_p.annotate("longest cointegrated\nregime", xy=(mid, 0.05), xytext=(mid, 0.3),
                       fontsize=8, ha="center", color="#2ca02c",
                       arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1))

    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)