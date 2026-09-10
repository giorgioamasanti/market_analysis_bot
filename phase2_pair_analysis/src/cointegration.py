"""
Core cointegration testing: Engle-Granger (pairs) and Johansen (N assets).

Both functions return a small dataclass rather than a tuple of raw numbers -
this matters once you're running hundreds of pairwise tests and rolling them
over time; named fields save you from off-by-one bugs reading tuple output.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# ---------------------------------------------------------------------------
# Engle-Granger (two assets)
# ---------------------------------------------------------------------------

@dataclass
class EngleGrangerResult:
    ticker_a: str
    ticker_b: str
    beta: float              # hedge ratio: units of B per unit of A
    alpha: float              # regression intercept
    residual: pd.Series       # the spread: A - alpha - beta * B
    coint_pvalue: float       # p-value of the EG cointegration test (adjusted crit. values)
    coint_tstat: float
    is_cointegrated: bool     # convenience flag at p < 0.05

    def __repr__(self):
        status = "COINTEGRATED" if self.is_cointegrated else "not cointegrated"
        return (f"EngleGranger({self.ticker_a}~{self.ticker_b}): "
                f"beta={self.beta:.3f}, p={self.coint_pvalue:.4f} [{status}]")


def engle_granger_test(price_a: pd.Series, price_b: pd.Series,
                        alpha_level: float = 0.05) -> EngleGrangerResult:
    """
    Standard two-step Engle-Granger test.

    Step 1: OLS regress A on B to get hedge ratio beta.
    Step 2: ADF-test the regression residual for stationarity, using
            statsmodels' `coint()`, which applies the correct (MacKinnon)
            critical values for a *residual from an estimated regression* -
            NOT the standard ADF critical values, which would be wrong here
            because beta itself was estimated from the same data.
    """
    a, b = price_a.align(price_b, join="inner")
    if len(a) < 30:
        raise ValueError(f"Only {len(a)} overlapping observations - need more history")

    # Step 1: hedge ratio via OLS (with intercept)
    X = sm.add_constant(b.values)
    model = sm.OLS(a.values, X).fit()
    alpha_const, beta = model.params
    residual = pd.Series(model.resid, index=a.index, name="spread")

    # Step 2: EG cointegration test (statsmodels does this correctly in one call -
    # it re-derives the right critical values internally rather than us
    # running a raw ADF on the residual with the wrong lookup table)
    eg_tstat, eg_pvalue, _ = coint(a.values, b.values)

    return EngleGrangerResult(
        ticker_a=price_a.name, ticker_b=price_b.name,
        beta=beta, alpha=alpha_const,
        residual=residual,
        coint_pvalue=eg_pvalue, coint_tstat=eg_tstat,
        is_cointegrated=eg_pvalue < alpha_level,
    )


def adf_pvalue(series: pd.Series) -> float:
    """Plain ADF test p-value - used to sanity-check that INPUT price series
    are individually non-stationary (I(1)) before you even bother testing
    for cointegration between them."""
    result = adfuller(series.dropna().values, autolag="AIC", result_object=False)
    return result[1]


# ---------------------------------------------------------------------------
# Johansen (N assets)
# ---------------------------------------------------------------------------

@dataclass
class JohansenResult:
    tickers: list[str]
    stable: bool                                  # False if the solve was numerically unstable
    rank: int | None = None                        # None when unstable - see fallback_pair instead
    trace_stats: np.ndarray | None = None
    trace_crit_95: np.ndarray | None = None
    eigenvectors: np.ndarray | None = None          # columns = cointegrating vectors, strongest first
    top_vector: np.ndarray | None = None
    fallback_pair: "EngleGrangerResult | None" = None  # auto-run when Johansen was unstable

    def __repr__(self):
        if not self.stable:
            return (f"Johansen({','.join(self.tickers)}): UNSTABLE (near-singular covariance matrix) "
                    f"- fell back to pairwise Engle-Granger on the offending pair: {self.fallback_pair}")
        return (f"Johansen({','.join(self.tickers)}): rank={self.rank}/{len(self.tickers)-1}, "
                f"top_vector={np.round(self.top_vector, 3).tolist()}")


import warnings
from numpy.exceptions import ComplexWarning


def _most_correlated_pair(prices: pd.DataFrame) -> tuple[str, str, float]:
    """Identifies which two columns are driving a collinearity problem -
    used both for the error/fallback message and to pick which pair to
    automatically re-test with Engle-Granger."""
    corr = prices.corr().abs()
    corr_vals = corr.values.copy()
    np.fill_diagonal(corr_vals, 0)
    i, j = np.unravel_index(np.argmax(corr_vals), corr_vals.shape)
    return corr.index[i], corr.columns[j], corr_vals[i, j]


def _run_johansen_checked(values: np.ndarray, det_order: int, k_ar_diff: int):
    """
    Runs coint_johansen while actively watching for the ComplexWarning that
    signals a near-singular covariance matrix. We deliberately do NOT rely
    on a correlation or condition-number threshold to predict this in
    advance - empirically, the failure doesn't track either cleanly (it
    also depends on det_order/lag structure), so guessing a cutoff either
    misses real failures or false-flags healthy, highly-correlated but
    genuinely cointegrated pairs. Catching the actual warning at the
    source is the only reliable signal.

    Returns (result, stable). When stable=False, the caller should not
    trust result at all - the reported eigenvalues/rank would be
    numerically meaningless (spurious complex values silently cast to
    real by statsmodels), not a genuine "no cointegration" finding.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = coint_johansen(values, det_order, k_ar_diff)
        stable = not any(issubclass(w.category, ComplexWarning) for w in caught)
    return result, stable


def johansen_test(prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1,
                   use_log: bool = True) -> JohansenResult:
    """
    Johansen trace test for cointegration rank among N price series.

    det_order=0: assumes no deterministic trend in the cointegrating
                 relationship (standard default for price-level data).
    k_ar_diff=1: number of lagged differences in the underlying VAR - 1 is
                 a reasonable default for daily data; increase if residuals
                 show autocorrelation.
    use_log=True: test on log prices rather than raw levels. This is
                 standard practice for Johansen - it stabilizes variance
                 across assets trading at very different price scales
                 (e.g. GLD ~$180 vs GDX ~$30), which otherwise adds its
                 own numerical conditioning problems on top of any real
                 collinearity between the series.

    Rank interpretation:
      0            -> no cointegration found among these assets
      1 .. N-1     -> that many independent stationary combinations exist
      N-1 (max)    -> as cointegrated as N assets can be

    We determine rank by comparing the trace statistic to the 95% critical
    value at each candidate rank, walking up from 0 - this is the standard
    sequential testing procedure (stop at the first rank you fail to reject).

    If the solve is numerically unstable (near-singular covariance matrix,
    almost always caused by two near-substitute assets in the group - e.g.
    GDX/GDXJ), we don't return a meaningless rank. Instead we automatically
    identify the most correlated pair and re-test THEM with a plain
    pairwise Engle-Granger test, since that's numerically robust regardless
    of how collinear two series are. Check `result.stable` - if False,
    `result.fallback_pair` holds that Engle-Granger result instead.
    """
    prices = prices.dropna()
    if len(prices) < 30:
        raise ValueError(f"Only {len(prices)} observations - need more history")

    if use_log and (prices.values <= 0).any():
        bad_cols = prices.columns[(prices <= 0).any()].tolist()
        raise ValueError(
            f"use_log=True requires strictly positive prices, but found non-positive "
            f"values in: {bad_cols}. Real market prices should never trigger this - "
            f"if you're testing with synthetic data, check your simulation doesn't "
            f"let a random walk drift below zero, or pass use_log=False."
        )

    values = np.log(prices.values) if use_log else prices.values
    raw_result, stable = _run_johansen_checked(values, det_order, k_ar_diff)

    if not stable:
        ticker_a, ticker_b, corr = _most_correlated_pair(prices)
        eg = engle_granger_test(prices[ticker_a], prices[ticker_b])
        return JohansenResult(
            tickers=list(prices.columns),
            stable=False,
            fallback_pair=eg,
        )

    trace_stats = raw_result.lr1            # trace statistic per candidate rank
    trace_crit_95 = raw_result.cvt[:, 1]    # 95% critical value column

    rank = 0
    for r in range(len(trace_stats)):
        if trace_stats[r] > trace_crit_95[r]:
            rank = r + 1
        else:
            break

    eigenvectors = raw_result.evec  # columns ordered by eigenvalue, strongest relationship first

    return JohansenResult(
        tickers=list(prices.columns),
        stable=True,
        rank=rank,
        trace_stats=trace_stats,
        trace_crit_95=trace_crit_95,
        eigenvectors=eigenvectors,
        top_vector=eigenvectors[:, 0],
    )