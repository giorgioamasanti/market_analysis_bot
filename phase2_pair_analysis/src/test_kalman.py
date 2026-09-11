"""
Validates kalman_hedge_ratio against synthetic data where the TRUE hedge
ratio is known and deliberately time-varying (a slow drift, then a sharper
step change) - checking that:

  1. The filter's estimated beta tracks the true trajectory reasonably
     closely (low mean absolute error).
  2. It genuinely adapts to the drift/step - unlike a single whole-sample
     OLS fit, which can only ever return ONE constant number and will be
     systematically wrong for most of the sample whenever the true beta
     is moving.

Run: python -m src.test_kalman
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.kalman import kalman_hedge_ratio

np.random.seed(21)
N = 1000


def make_time_varying_pair():
    """B is a random walk. A tracks B with a beta that DRIFTS smoothly
    from 1.0 to 2.0 over the first 600 days, then STEP-CHANGES to 0.5 for
    the remaining 400 - a drift regime followed by an abrupt regime, to
    stress-test both modes of adaptation."""
    b = 100 + np.cumsum(np.random.normal(0, 1, N))

    true_beta = np.empty(N)
    true_beta[:600] = np.linspace(1.0, 2.0, 600)   # smooth drift
    true_beta[600:] = 0.5                          # abrupt step change

    stationary_noise = np.zeros(N)
    for t in range(1, N):
        stationary_noise[t] = 0.7 * stationary_noise[t - 1] + np.random.normal(0, 1)

    a = 20 + true_beta * b + stationary_noise

    idx = pd.date_range("2020-01-01", periods=N, freq="B")
    return (pd.Series(a, index=idx, name="A"),
            pd.Series(b, index=idx, name="B"),
            pd.Series(true_beta, index=idx, name="true_beta"))


def run():
    a, b, true_beta = make_time_varying_pair()

    result = kalman_hedge_ratio(a, b, delta=1e-4)
    print(result)

    # skip the burn-in period for error measurement - the filter hasn't
    # had a chance to converge yet in the first ~30 obs, that's expected
    burn = 30
    mae = np.abs(result.beta.iloc[burn:] - true_beta.iloc[burn:]).mean()
    print(f"\nMean absolute error vs true beta (post burn-in): {mae:.4f}")

    # compare against a single whole-sample OLS beta - the thing the
    # Kalman filter is supposed to improve on
    X = sm.add_constant(b.values)
    ols_beta = sm.OLS(a.values, X).fit().params[1]
    ols_mae = np.abs(ols_beta - true_beta.iloc[burn:]).mean()
    print(f"Static whole-sample OLS beta: {ols_beta:.4f} "
          f"(mean absolute error vs true beta: {ols_mae:.4f})")

    assert mae < ols_mae, "FAIL: Kalman filter should track true beta better than a single static OLS fit"
    print("\n-> Kalman filter tracks the time-varying true beta substantially "
          "better than a single static fit, as expected")

    # check specifically that it adapts to the step change: beta near the
    # end should be much closer to 0.5 (post-step) than to 2.0 (pre-step)
    final_beta = result.beta.iloc[-30:].mean()
    print(f"\nMean beta over final 30 days: {final_beta:.3f} "
          f"(true post-step value: 0.5)")
    assert abs(final_beta - 0.5) < 0.3, "FAIL: filter did not adapt to the step change"
    print("-> correctly adapted to the abrupt regime change\n")

    print("=" * 70)
    print("ALL KALMAN FILTER CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run()