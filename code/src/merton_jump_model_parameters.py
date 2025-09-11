import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm, probplot

# ----------------------------
# Merton MLE
# ----------------------------
@dataclass
class MertonParams:
    alpha: float   # drift in log-return eq (pre-compensation form)
    sigma: float   # diffusion vol
    lam: float     # jump intensity (per year)
    m: float       # mean log-jump
    s: float       # std log-jump
    mu: float      # implied drift of S_t under P (for reference)

def _neg_loglike_merton(params_vec: np.ndarray, r: np.ndarray, dt: float, n_max: int) -> float:
    """
    params_vec = [alpha, sigma, lam, m, s] with bounds sigma>0, lam>=0, s>0 (enforced by optimizer).
    Returns negative log-likelihood for independent increments.
    """
    alpha, sigma, lam, m, s = params_vec
    if sigma <= 0 or lam < 0 or s <= 0:
        return np.inf

    # Precompute Poisson log-weights for 0..n_max
    # log w_n = log( e^{-lam dt} (lam dt)^n / n! )
    # Compute in log to avoid under/overflow
    from math import lgamma
    log_pois = np.array([ -lam*dt + n*np.log(lam*dt) - lgamma(n+1) if lam*dt > 0 else (0.0 if n==0 else -np.inf)
                          for n in range(n_max+1) ])

    # For each n, NormalPDF(r; mean=alpha*dt + n*m, var = sigma^2*dt + n*s^2)
    # Work in log domain: log φ = -0.5*log(2π) -0.5*log(var) - (r-mean)^2/(2 var)
    r = r.reshape(-1, 1)  # (T,1)
    ns = np.arange(0, n_max+1).reshape(1, -1)  # (1, N)
    means = alpha*dt + ns * m
    vars_ = (sigma**2)*dt + ns * (s**2)
    # Guard against zero variance at n=0 and tiny dt
    vars_ = np.maximum(vars_, 1e-18)

    log_norm = -0.5*np.log(2*np.pi) - 0.5*np.log(vars_) - (r - means)**2 / (2*vars_)
    # mixture: log f_t = logsumexp( log_pois + log_norm_n )
    log_weights = log_pois.reshape(1, -1)
    log_mix = logsumexp(log_weights + log_norm, axis=1)
    nll = -np.sum(log_mix)
    return float(nll)

def fit_merton_mle(r: np.ndarray, dt: float, n_max: int = 5) -> MertonParams:
    """
    MLE for Merton on log-returns r (1D array). Returns MertonParams.
    """
    # Initial guesses
    std_r = np.std(r, ddof=1)
    sigma0 = max(std_r / np.sqrt(dt), 1e-4)    # crude: attribute all var to diffusion initially
    lam0   = 80.0                              # per year (≈0.22/day). Adjust if desired.
    m0     = 0.0
    s0     = 0.02
    # alpha ≈ E[r]/dt - lam*m  (since E[r] = α dt + lam m dt)
    alpha0 = float(np.mean(r)/dt - lam0*m0)

    x0 = np.array([alpha0, sigma0, lam0, m0, s0], dtype=float)
    bounds = [(-5.0, 5.0),       # alpha (per year)
              (1e-6, 5.0),      # sigma (per sqrt(year))
              (1e-8, 500.0),    # lam (per year)
              (-1.0, 1.0),      # m (log-jump mean)
              (1e-6, 1.0)]      # s (log-jump std)

    res = minimize(_neg_loglike_merton, x0,
                   args=(r, dt, n_max),
                   method="L-BFGS-B",
                   bounds=bounds,
                   options=dict(maxiter=500, ftol=1e-9))

    if not res.success:
        print("[WARN] MLE did not fully converge:", res.message)

    alpha, sigma, lam, m, s = res.x
    # implied μ for S_t (helpful for simulation under P if you use dS/S form)
    mu = alpha + 0.5*sigma**2 + lam*(np.exp(m + 0.5*s**2) - 1.0)
    return MertonParams(alpha=float(alpha), sigma=float(sigma), lam=float(lam),
                        m=float(m), s=float(s), mu=float(mu))
