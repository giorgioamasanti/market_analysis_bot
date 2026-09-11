"""
Rolls Engle-Granger over a moving window through time, so instead of one
static "cointegrated: yes/no" answer, you get a *time series* of p-values
and hedge ratios - this is what lets you actually see a relationship
strengthen, weaken, or break down, rather than assuming it's constant.

This is deliberately kept simple (rolling OLS + rolling ADF) rather than
using the Kalman filter yet - that comes next. Think of this as the
"coarse, robust" view of drift, and the Kalman filter as the "smooth,
continuous" view. Comparing the two is a good cross-check in itself.
"""

import numpy as np
import pandas as pd

from src.cointegration import engle_granger_test


def rolling_engle_granger(price_a: pd.Series, price_b: pd.Series,
                           window: int = 252, step: int = 5) -> pd.DataFrame:
    """
    Re-runs Engle-Granger on every trailing `window`-day slice, stepping
    forward `step` days at a time (step > 1 just for speed - daily is
    usually unnecessary given a 252-day window barely changes day to day).

    window=252 (~1 trading year) is a reasonable default: long enough for
    the ADF test to have power, short enough to actually detect a regime
    change within a year rather than averaging over multiple regimes.

    Returns a DataFrame indexed by the END date of each window, with
    columns: pvalue, beta, is_cointegrated.
    """
    a, b = price_a.align(price_b, join="inner")
    dates, pvalues, betas, cointegrated = [], [], [], []

    for end in range(window, len(a) + 1, step):
        start = end - window
        window_a = a.iloc[start:end]
        window_b = b.iloc[start:end]

        try:
            result = engle_granger_test(window_a, window_b)
            dates.append(a.index[end - 1])
            pvalues.append(result.coint_pvalue)
            betas.append(result.beta)
            cointegrated.append(result.is_cointegrated)
        except Exception:
            continue  # skip windows that fail (e.g. too much missing data)

    return pd.DataFrame({
        "pvalue": pvalues,
        "beta": betas,
        "is_cointegrated": cointegrated,
    }, index=pd.DatetimeIndex(dates, name="window_end"))


def summarize_stability(rolling_result: pd.DataFrame) -> dict:
    """
    Turns the rolling output into a few numbers that describe HOW STABLE
    the relationship is - this is the actual research deliverable for
    phase 2's "where does it break down" question.
    """
    pct_time_cointegrated = 100 * rolling_result["is_cointegrated"].mean()
    beta_std = rolling_result["beta"].std()
    beta_range = rolling_result["beta"].max() - rolling_result["beta"].min()
    beta_mean = rolling_result["beta"].mean()
    # coefficient of variation: beta_std relative to beta's own magnitude -
    # a hedge ratio of "1.09 +/- 0.0007" is trivially stable (CV ~0.06%), a
    # hedge ratio of "1.5 +/- 0.6" (CV ~40%) is not really a fixed ratio at
    # all, regardless of what the significance test says about any single window
    beta_cv_pct = 100 * beta_std / abs(beta_mean) if beta_mean != 0 else float("inf")

    # Longest consecutive stretch NOT cointegrated - a single bad week is
    # noise, a 6-month stretch of non-cointegration is a genuine breakdown.
    not_coint = ~rolling_result["is_cointegrated"]
    groups = (not_coint != not_coint.shift()).cumsum()
    run_lengths = not_coint.groupby(groups).sum()
    longest_breakdown = int(run_lengths.max()) if len(run_lengths) else 0

    return {
        "pct_time_cointegrated": round(pct_time_cointegrated, 1),
        "beta_std": round(beta_std, 4),
        "beta_range": round(beta_range, 4),
        "beta_cv_pct": round(beta_cv_pct, 2),
        "longest_breakdown_windows": longest_breakdown,
    }


def identify_regimes(rolling_result: pd.DataFrame) -> pd.DataFrame:
    """
    Groups the rolling output into contiguous "regimes" - maximal runs of
    consecutive windows sharing the same is_cointegrated value - instead of
    a long list of individual window flips. This is what actually answers
    "where should I look" rather than making you eyeball a transition list.

    Returns one row per regime: start/end date, duration, mean p-value and
    beta within it. A regime with is_cointegrated=True and very low mean
    p-value spanning a long duration is a strong "study/trade this window"
    candidate; a long is_cointegrated=False regime is a genuine breakdown
    period worth understanding (see summarize_regimes for auto-highlights).
    """
    grouped = rolling_result.copy()
    # each time is_cointegrated flips, start a new group id
    group_id = (grouped["is_cointegrated"] != grouped["is_cointegrated"].shift()).cumsum()

    regimes = []
    for _, block in grouped.groupby(group_id):
        regimes.append({
            "start": block.index[0],
            "end": block.index[-1],
            "is_cointegrated": bool(block["is_cointegrated"].iloc[0]),
            "n_windows": len(block),
            "mean_pvalue": block["pvalue"].mean(),
            "min_pvalue": block["pvalue"].min(),
            "mean_beta": block["beta"].mean(),
            "beta_std": block["beta"].std() if len(block) > 1 else 0.0,
        })

    return pd.DataFrame(regimes)


def summarize_regimes(regimes: pd.DataFrame, top_n: int = 3) -> str:
    """
    Picks out the regimes actually worth looking at by hand, rather than
    making you scan the full table:
      - longest cointegrated regime(s) - best candidate windows to study
        or build a strategy around
      - longest breakdown regime(s) - the most significant instability,
        worth understanding WHY (a specific event? a slow structural drift?)
      - strongest cointegrated regime(s) by lowest mean p-value - most
        statistically confident, even if not the longest

    Returns a formatted string ready to print.
    """
    coint = regimes[regimes["is_cointegrated"]].sort_values("n_windows", ascending=False)
    breakdown = regimes[~regimes["is_cointegrated"]].sort_values("n_windows", ascending=False)
    strongest = regimes[regimes["is_cointegrated"]].sort_values("mean_pvalue")

    lines = []

    lines.append("Longest cointegrated regime(s) - best windows to study/trade:")
    if coint.empty:
        lines.append("  (none - never cointegrated in this sample)")
    for _, r in coint.head(top_n).iterrows():
        lines.append(f"  {r['start'].date()} -> {r['end'].date()}  "
                      f"({r['n_windows']} windows, mean p={r['mean_pvalue']:.4f}, "
                      f"beta={r['mean_beta']:.3f} +/- {r['beta_std']:.4f})")

    lines.append("\nLongest breakdown regime(s) - most significant instability:")
    for _, r in breakdown.head(top_n).iterrows():
        lines.append(f"  {r['start'].date()} -> {r['end'].date()}  ({r['n_windows']} windows)")

    lines.append("\nMost confidently cointegrated regime(s) - lowest average p-value:")
    if strongest.empty:
        lines.append("  (none)")
    for _, r in strongest.head(top_n).iterrows():
        lines.append(f"  {r['start'].date()} -> {r['end'].date()}  "
                      f"(mean p={r['mean_pvalue']:.4f}, {r['n_windows']} windows)")

    return "\n".join(lines)