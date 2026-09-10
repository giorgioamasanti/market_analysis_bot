"""
Sanity-checks cointegration.py against synthetic data where we KNOW the
true answer, before trusting it on real prices:

  1. Two genuinely cointegrated series (constructed by design) -> EG and
     Johansen should both confidently detect cointegration.
  2. Two independent random walks -> should NOT find cointegration
     (this is the classic "spurious regression" trap - two unrelated
     random walks will often show high R^2 and look correlated, but
     should fail a proper cointegration test).

Run: python -m src.test_cointegration
"""

import numpy as np
import pandas as pd

from src.cointegration import engle_granger_test, johansen_test, adf_pvalue

np.random.seed(42)
N = 1000


def make_cointegrated_pair():
    """Construct A and B that are cointegrated by design: B is a random
    walk, A tracks B plus a stationary (mean-reverting) noise term. The
    combination (A - 2*B) is stationary by construction."""
    common_walk = np.cumsum(np.random.normal(0, 1, N))
    b = 100 + common_walk
    stationary_noise = np.zeros(N)
    for t in range(1, N):
        stationary_noise[t] = 0.8 * stationary_noise[t - 1] + np.random.normal(0, 1)
    a = 50 + 2 * common_walk + stationary_noise  # true beta = 2

    idx = pd.date_range("2020-01-01", periods=N, freq="B")
    return pd.Series(a, index=idx, name="A"), pd.Series(b, index=idx, name="B")


def make_independent_walks():
    """Two totally unrelated random walks - classic spurious regression
    trap. A naive OLS/correlation will often look convincing; a correct
    cointegration test should NOT be fooled."""
    a = 100 + np.cumsum(np.random.normal(0, 1, N))
    b = 100 + np.cumsum(np.random.normal(0, 1, N))
    idx = pd.date_range("2020-01-01", periods=N, freq="B")
    return pd.Series(a, index=idx, name="X"), pd.Series(b, index=idx, name="Y")


def test_johansen_fallback():
    """
    Validates the automatic Engle-Granger fallback when Johansen hits a
    near-singular covariance matrix. We force the unstable code path
    directly (via monkeypatching the internal solve) rather than relying
    on a random synthetic draw to happen to trigger the real statsmodels
    ComplexWarning - that's possible but not reliably reproducible on
    demand, since it depends on internals we don't control. This checks
    the fallback WIRING is correct regardless of what triggers it.
    """
    from unittest.mock import patch

    idx = pd.date_range("2022-01-01", periods=200, freq="B")
    a = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 200)), index=idx, name="A")
    b = (a * 1.0001).rename("B")  # deliberate near-duplicate, so we can check
                                    # the fallback picks THIS pair specifically
    c = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 200)), index=idx, name="C")
    df = pd.DataFrame({"A": a, "B": b, "C": c})

    def fake_unstable(*args, **kwargs):
        return None, False  # simulate statsmodels reporting instability

    with patch("src.cointegration._run_johansen_checked", side_effect=fake_unstable):
        result = johansen_test(df)

    assert result.stable is False, "FAIL: should report instability"
    assert result.fallback_pair is not None, "FAIL: should have run the EG fallback"
    assert {result.fallback_pair.ticker_a, result.fallback_pair.ticker_b} == {"A", "B"}, \
        "FAIL: fallback should target the most correlated pair (A, B)"
    print(f"Johansen fallback correctly identified and re-tested the offending pair: "
          f"{result.fallback_pair}")
    print("  -> automatic fallback wiring confirmed correct\n")


def run():
    print("=" * 70)
    print("TEST 1: genuinely cointegrated pair (true beta = 2)")
    print("=" * 70)
    a, b = make_cointegrated_pair()

    print(f"ADF p-value on raw A: {adf_pvalue(a):.4f}  (expect NON-stationary, p > 0.05)")
    print(f"ADF p-value on raw B: {adf_pvalue(b):.4f}  (expect NON-stationary, p > 0.05)")

    eg = engle_granger_test(a, b)
    print(eg)
    assert eg.is_cointegrated, "FAIL: should have detected cointegration"
    assert abs(eg.beta - 2.0) < 0.15, f"FAIL: recovered beta {eg.beta:.3f} far from true beta 2.0"
    print("  -> correctly detected cointegration, beta recovered accurately\n")

    jh = johansen_test(pd.DataFrame({"A": a, "B": b}))
    print(jh)
    assert jh.rank >= 1, "FAIL: Johansen should find rank >= 1"
    print("  -> Johansen agrees\n")

    print("=" * 70)
    print("TEST 2: two independent random walks (spurious regression trap)")
    print("=" * 70)
    x, y = make_independent_walks()

    naive_corr = np.corrcoef(x, y)[0, 1]
    print(f"Naive price correlation: {naive_corr:.3f}  <- looks deceptively high, this is the trap")

    eg2 = engle_granger_test(x, y)
    print(eg2)
    assert not eg2.is_cointegrated, "FAIL: should NOT have detected cointegration in independent walks"
    print("  -> correctly rejected cointegration despite high naive correlation\n")

    jh2 = johansen_test(pd.DataFrame({"X": x, "Y": y}))
    print(jh2)
    assert jh2.rank == 0, "FAIL: Johansen should find rank 0 for independent walks"
    print("  -> Johansen agrees\n")

    print("=" * 70)
    print("ALL CHECKS PASSED - cointegration.py is behaving correctly")
    print("=" * 70)


if __name__ == "__main__":
    run()
    print()
    test_johansen_fallback()