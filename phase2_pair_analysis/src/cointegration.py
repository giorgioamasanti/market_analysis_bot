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
    rank: int                       # number of cointegrating relationships found
    trace_stats: np.ndarray
    trace_crit_95: np.ndarray
    eigenvectors: np.ndarray        # columns = cointegrating vectors (beta), strongest first
    top_vector: np.ndarray          # the single strongest cointegrating relationship

    def __repr__(self):
        return (f"Johansen({','.join(self.tickers)}): rank={self.rank}/{len(self.tickers)-1}, "
                f"top_vector={np.round(self.top_vector, 3).tolist()}")


import warnings
from numpy.exceptions import ComplexWarning


def _pairwise_corr_report(prices: pd.DataFrame) -> str:
    """Human-readable pairwise correlations, for the error message below -
    tells you WHICH pair to investigate/drop without having to go look."""
    corr = prices.corr().abs()
    corr_vals = corr.values.copy()
    np.fill_diagonal(corr_vals, 0)
    i, j = np.unravel_index(np.argmax(corr_vals), corr_vals.shape)
    return f"{corr.index[i]}~{corr.columns[j]} at {corr_vals[i, j]:.4f}"


def _run_johansen_checked(values: np.ndarray, det_order: int, k_ar_diff: int,
                           tickers: list[str], prices: pd.DataFrame):
    """
    Runs coint_johansen while actively watching for the ComplexWarning that
    signals a near-singular covariance matrix. We deliberately do NOT rely
    on a correlation or condition-number threshold to predict this in
    advance - empirically, the failure doesn't track either cleanly (it
    also depends on det_order/lag structure), so guessing a cutoff either
    misses real failures or false-flags healthy, highly-correlated but
    genuinely cointegrated pairs. Catching the actual warning at the
    source is the only reliable signal.

    If it fires, the reported rank/eigenvectors are numerically meaningless
    (spurious complex values silently cast to real) and we raise rather
    than let a broken result look like a real "no cointegration" finding.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = coint_johansen(values, det_order, k_ar_diff)

        if any(issubclass(w.category, ComplexWarning) for w in caught):
            raise ValueError(
                f"Johansen test on {tickers} hit a near-singular covariance matrix "
                f"(statsmodels had to discard complex eigenvalue components - the "
                f"reported rank would be numerically meaningless, not a real finding "
                f"of 'no cointegration'). Most correlated pair in this group: "
                f"{_pairwise_corr_report(prices)}. These are likely near-substitutes - "
                f"either test the group's pairs individually with engle_granger_test(), "
                f"or drop one of the two most-correlated assets and retry Johansen."
            )

    return result


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

    Raises ValueError if any two input series are too collinear for a
    numerically stable solve (see _check_collinearity) - this is a real
    failure mode with near-substitute assets (e.g. GDX/GDXJ), not an edge
    case worth ignoring: an ill-conditioned solve can silently produce
    complex eigenvalues that get cast to real, making the reported rank
    meaningless rather than merely imprecise.
    """
    prices = prices.dropna()
    if len(prices) < 30:
        raise ValueError(f"Only {len(prices)} observations - need more history")

    values = np.log(prices.values) if use_log else prices.values
    result = _run_johansen_checked(values, det_order, k_ar_diff, list(prices.columns), prices)

    trace_stats = result.lr1            # trace statistic per candidate rank
    trace_crit_95 = result.cvt[:, 1]    # 95% critical value column

    rank = 0
    for r in range(len(trace_stats)):
        if trace_stats[r] > trace_crit_95[r]:
            rank = r + 1
        else:
            break

    eigenvectors = result.evec  # columns ordered by eigenvalue, strongest relationship first

    return JohansenResult(
        tickers=list(prices.columns),
        rank=rank,
        trace_stats=trace_stats,
        trace_crit_95=trace_crit_95,
        eigenvectors=eigenvectors,
        top_vector=eigenvectors[:, 0],
    )