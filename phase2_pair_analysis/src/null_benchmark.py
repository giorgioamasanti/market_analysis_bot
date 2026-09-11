"""
Automatic significance benchmark for rolling cointegration results.

The core idea (validated manually in the SPY-SMH investigation): "% of
windows flagged cointegrated" is meaningless on its own - you need to know
what that number would look like for two assets with NO real relationship,
tested with the exact same window/step/sample-length. This module makes
that comparison automatic instead of a one-off simulation.

Usage:
    from src.rolling_analysis import rolling_engle_granger
    from src.null_benchmark import benchmark_against_null

    rolling = rolling_engle_granger(prices['GDX'], prices['GDXJ'])
    result = benchmark_against_null(rolling)
    print(result)
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from src.rolling_analysis import rolling_engle_granger


@dataclass
class NullBenchmarkResult:
    observed_pct: float          # actual % of windows flagged cointegrated
    null_mean: float             # mean % flagged, across simulated unrelated-asset trials
    null_std: float
    null_p95: float              # 95th percentile of the null distribution -
                                  # a rough "you'd need to beat this to look
                                  # different from noise" bar
    n_trials: int
    z_score: float                # (observed - null_mean) / null_std
    verdict: str

    def __repr__(self):
        return (f"NullBenchmark: observed={self.observed_pct:.1f}% vs "
                f"null={self.null_mean:.1f}% +/- {self.null_std:.1f}% "
                f"(95th pctile={self.null_p95:.1f}%), z={self.z_score:.2f} "
                f"-> {self.verdict}")


@lru_cache(maxsize=64)
def _simulate_null_pcts(n_days: int, window: int, step: int,
                         n_trials: int, seed: int) -> tuple[float, ...]:
    """
    Cached simulation: for a given sample length/window/step, runs the
    rolling Engle-Granger test on `n_trials` pairs of INDEPENDENT random
    walks and returns the % of windows each falsely flags as cointegrated.

    Cached (keyed on the exact config) because this is the expensive part -
    once you've simulated the null for window=252/step=5/n_days=2273, you
    don't want to re-run it for every pair you benchmark against the same
    setup. Returns a tuple (not a list/array) specifically so it's hashable
    and lru_cache-able.
    """
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2000-01-01", periods=n_days, freq="B")
    pcts = []

    for _ in range(n_trials):
        a = pd.Series(100 + np.cumsum(rng.normal(0, 1, n_days)), index=idx, name="A")
        b = pd.Series(100 + np.cumsum(rng.normal(0, 1, n_days)), index=idx, name="B")
        roll = rolling_engle_granger(a, b, window=window, step=step)
        pcts.append(100 * roll["is_cointegrated"].mean())

    return tuple(pcts)


def benchmark_against_null(rolling_result: pd.DataFrame, window: int = 252,
                            step: int = 5, n_trials: int = 20,
                            seed: int = 0) -> NullBenchmarkResult:
    """
    Benchmarks an existing rolling_engle_granger() result against a
    simulated null of unrelated random walks, matched on sample length,
    window, and step so the comparison is apples-to-apples.

    window/step MUST match what was used to produce `rolling_result` - if
    you rolled with window=126, pass window=126 here too, or the null
    simulation won't be a fair comparison.
    """
    observed_pct = 100 * rolling_result["is_cointegrated"].mean()

    n_windows = len(rolling_result)
    # invert the rolling_engle_granger windowing math to recover roughly
    # how many raw days of data produced this many windows
    n_days = window + (n_windows - 1) * step

    null_pcts = np.array(_simulate_null_pcts(n_days, window, step, n_trials, seed))
    null_mean = null_pcts.mean()
    null_std = null_pcts.std()
    null_p95 = np.percentile(null_pcts, 95)

    z = (observed_pct - null_mean) / null_std if null_std > 0 else np.inf

    if observed_pct <= null_p95:
        verdict = "NOT distinguishable from noise - treat with real suspicion"
    elif z < 3:
        verdict = "above the null, but only modestly - worth corroborating (beta stability, rolling transitions)"
    else:
        verdict = "clearly beyond what unrelated assets produce - likely a genuine relationship"

    return NullBenchmarkResult(
        observed_pct=observed_pct, null_mean=null_mean, null_std=null_std,
        null_p95=null_p95, n_trials=n_trials, z_score=z, verdict=verdict,
    )