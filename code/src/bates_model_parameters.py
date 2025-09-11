import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
from math import lgamma
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm, probplot

import src.heston_model_parameters as hm  # your previous Heston estimator

@dataclass
class BatesParams:
    # Heston part (from hm estimator)
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float
    mu: float        # drift of S under P (real-world)
    lam: float       # jump intensity (per year)
    m: float         # mean log-jump
    s: float         # std log-jump

def neg_loglike_bates_jumps(params_vec: np.ndarray,
                            r: np.ndarray,
                            vhat_tm1: np.ndarray,
                            dt: float,
                            n_max: int) -> float:
    """
    Conditional mixture likelihood for Bates given v_{t-1} proxy:
    r_t | v_{t-1} ~ sum_{n>=0} Pois(n; λdt) * N( mean_t + n m, var_t + n s^2 )
    where mean_t = (μ - λk - 0.5 v_{t-1}) dt, var_t = v_{t-1} dt, k = exp(m + 0.5 s^2) - 1
    params_vec = [mu, lam, m, s], with lam>=0, s>0.
    """
    mu, lam, m, s = params_vec
    if lam < 0 or s <= 0:
        return np.inf

    k = np.exp(m + 0.5*s*s) - 1.0

    mean_base = (mu - lam * k - 0.5 * vhat_tm1) * dt   # shape (T,)
    var_base  = vhat_tm1 * dt                          # shape (T,)
    var_base  = np.maximum(var_base, 1e-18)

    # Poisson log-weights for n=0..n_max
    if lam * dt > 0:
        n = np.arange(n_max + 1)
        log_pois = (-lam*dt) + n * np.log(lam*dt) - np.array([lgamma(ni+1) for ni in n])
    else:
        log_pois = np.array([0.0] + [-np.inf]*n_max)

    # Build (T, n_max+1) grids
    T = r.shape[0]
    ns = np.arange(0, n_max+1).reshape(1, -1)                   # (1,N)
    means = mean_base.reshape(-1,1) + ns * m                    # (T,N)
    vars_ = var_base.reshape(-1,1) + ns * (s*s)                 # (T,N)
    vars_ = np.maximum(vars_, 1e-18)

    log_norm = -0.5*np.log(2*np.pi) - 0.5*np.log(vars_) - (r.reshape(-1,1) - means)**2/(2*vars_)
    log_mix  = logsumexp(log_norm + log_pois.reshape(1,-1), axis=1)  # (T,)
    nll = -float(np.sum(log_mix))
    return nll

def fit_bates_jumps_mle(
    r_train: pd.Series,
    vhat_train: pd.Series,
    dt: float,
    n_max: int = 7,
    *,
    # New: bound kwargs (low, high)
    bounds_mu: tuple = (-5.0, 5.0),            # μ per year
    bounds_lambda: tuple = (1e-8, 500.0),      # λ per year
    bounds_m: tuple = (-1.0, 1.0),             # mean log-jump
    bounds_s: tuple = (1e-6, 1.0),             # std log-jump
    # Optional: initial guess + optimizer options
    x0: np.ndarray | None = None,
    opt_options: dict | None = None,
) -> Tuple[float, float, float, float]:
    """
    Fit (mu, lam, m, s) by MLE on training returns given vhat_{t-1}.
    You can constrain each parameter via bounds_* kwargs.

    Parameters
    ----------
    r_train : pd.Series
        Log returns at 1m frequency (or consistent with dt).
    vhat_train : pd.Series
        Variance proxy (annualized), aligned to r_train timestamps.
    dt : float
        Step size in years (e.g., 1/(365*24*60) for 1-minute).
    n_max : int
        Poisson truncation in the likelihood.
    bounds_* : tuple(low, high)
        Bounds for μ, λ, m, s.
    x0 : np.ndarray or None
        Optional initial guess [mu, lam, m, s]. If None, sensible defaults used.
    opt_options : dict or None
        Passed to scipy.optimize.minimize (e.g., {'maxiter':600, 'ftol':1e-9}).

    Returns
    -------
    (mu, lam, m, s) : tuple of floats
    """
    vhat_tm1 = vhat_train.shift(1)
    idx = r_train.index.intersection(vhat_tm1.dropna().index)
    r = r_train.loc[idx].values
    v_tm1 = vhat_tm1.loc[idx].values

    # Initial guesses (if not provided)
    if x0 is None:
        m0, s0 = 0.0, 0.02
        lam0   = min(80.0, bounds_lambda[1])  # respect upper bound
        mu0    = float(r.mean() / dt + 0.5 * np.mean(v_tm1))
        x0 = np.array([mu0, lam0, m0, s0], dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float)

    bounds = [bounds_mu, bounds_lambda, bounds_m, bounds_s]

    if opt_options is None:
        opt_options = dict(maxiter=600, ftol=1e-9)

    res = minimize(
        neg_loglike_bates_jumps,
        x0,
        args=(r, v_tm1, dt, n_max),
        method="L-BFGS-B",
        bounds=bounds,
        options=opt_options,
    )
    if not res.success:
        print("[WARN] Bates MLE did not fully converge:", res.message)

    mu, lam, m, s = res.x
    return float(mu), float(lam), float(m), float(s)