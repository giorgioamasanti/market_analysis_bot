"""
Kalman filter for a time-varying hedge ratio, following the standard
2-state (alpha, beta) formulation used in pairs trading (see Ernie Chan,
"Algorithmic Trading", ch. 3 - this is the same construction).

Why this instead of rolling OLS (rolling_analysis.py):
  - Rolling OLS re-estimates from scratch on a fixed window, discards
    everything outside it, and only updates every `step` days - it reacts
    to change in lagged, chunky jumps and requires choosing an arbitrary
    window length.
  - The Kalman filter updates continuously, using ALL history (weighted
    by recency via the Kalman gain, not a hard window cutoff), and gives
    you a genuine real-time estimate usable for actual trading, not just
    diagnostics.

State-space model:
    x_t = [alpha_t, beta_t]'            (hidden state: intercept + hedge ratio)
    x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)  (state evolves as a random walk)
    y_t = H_t x_t + eps_t, eps_t ~ N(0, R) (observation: y_t = price_A(t),
                                            H_t = [1, price_B(t)])

The innovation e_t = y_t - H_t @ x_pred IS the live spread you'd trade -
it's the model's real-time prediction error, updated fresh every day.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KalmanHedgeResult:
    alpha: pd.Series        # time-varying intercept
    beta: pd.Series         # time-varying hedge ratio - the main output
    spread: pd.Series       # innovation e_t: the live, tradeable spread
    spread_std: pd.Series   # sqrt(S_t): innovation std dev at each step
    zscore: pd.Series       # spread / spread_std - ready to threshold for signals

    def __repr__(self):
        return (f"KalmanHedgeResult(n={len(self.beta)}, "
                f"beta: {self.beta.iloc[0]:.3f} -> {self.beta.iloc[-1]:.3f}, "
                f"final zscore={self.zscore.iloc[-1]:.2f})")


def kalman_hedge_ratio(price_a: pd.Series, price_b: pd.Series,
                        delta: float = 1e-4,
                        obs_covariance: float | None = None,
                        burn_in: int = 30) -> KalmanHedgeResult:
    """
    Runs the Kalman filter over aligned price series and returns the full
    time series of hedge ratios, live spread, and z-score.

    delta: controls how fast beta is ALLOWED to drift. Sets the state
           noise covariance Q = delta/(1-delta) * I(2). Smaller delta ->
           smoother, slower-adapting beta (closer to a static OLS estimate
           over the whole history). Larger delta -> beta reacts faster to
           recent data, at the cost of more noise in the estimate. 1e-4 to
           1e-5 is a typical starting range for daily data - tune it by
           checking (as we do in the validation below) whether the filter
           tracks a KNOWN synthetic beta trajectory well.

    obs_covariance: R, the assumed noise variance on the price observation
           itself. If None, estimated from the residual variance of a
           simple OLS fit on the first `burn_in` observations - a
           reasonable default that adapts to the actual noise level of
           the pair rather than requiring you to guess a number upfront.

    burn_in: number of initial observations used only to (a) estimate R
           if not provided, and (b) initialize alpha/beta via plain OLS,
           rather than starting the filter cold at [0, 0] and wasting the
           first stretch of data on convergence.
    """
    a, b = price_a.align(price_b, join="inner")
    n = len(a)
    if n < burn_in + 30:
        raise ValueError(f"Only {n} observations - need at least {burn_in + 30}")

    # --- initialize from a burn-in OLS fit, rather than starting blind ---
    import statsmodels.api as sm
    X0 = sm.add_constant(b.iloc[:burn_in].values)
    ols = sm.OLS(a.iloc[:burn_in].values, X0).fit()
    alpha0, beta0 = ols.params

    if obs_covariance is None:
        obs_covariance = float(np.var(ols.resid))
    R = obs_covariance

    Q = delta / (1 - delta) * np.eye(2)  # state (process) noise covariance

    # state and its uncertainty
    x = np.array([alpha0, beta0])
    P = np.eye(2) * 1.0  # initial state uncertainty - fairly uninformative

    alphas, betas, spreads, spread_stds = [], [], [], []

    for t in range(n):
        y_t = a.iloc[t]
        H_t = np.array([1.0, b.iloc[t]])  # observation vector: [1, price_B]

        # --- predict step ---
        # state transition is identity (random walk), so x_pred = x, but
        # uncertainty grows by Q every step
        x_pred = x
        P_pred = P + Q

        # --- update step ---
        y_hat = H_t @ x_pred                    # predicted price_A
        e_t = y_t - y_hat                        # innovation = live spread
        S_t = H_t @ P_pred @ H_t.T + R            # innovation variance
        K_t = P_pred @ H_t.T / S_t                # Kalman gain (2,)

        x = x_pred + K_t * e_t
        P = P_pred - np.outer(K_t, H_t) @ P_pred

        alphas.append(x[0])
        betas.append(x[1])
        spreads.append(e_t)
        spread_stds.append(np.sqrt(S_t))

    idx = a.index
    spread = pd.Series(spreads, index=idx, name="spread")
    spread_std = pd.Series(spread_stds, index=idx, name="spread_std")
    zscore = (spread / spread_std).rename("zscore")

    return KalmanHedgeResult(
        alpha=pd.Series(alphas, index=idx, name="alpha"),
        beta=pd.Series(betas, index=idx, name="beta"),
        spread=spread,
        spread_std=spread_std,
        zscore=zscore,
    )