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
        "longest_breakdown_windows": longest_breakdown,
    }