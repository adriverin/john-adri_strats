import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple
from scipy.optimize import minimize

@dataclass
class HestonParams:
    mu: float
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float

def log_returns(prices: pd.Series) -> pd.Series:
    r = np.log(prices / prices.shift(1)).dropna()
    r.name = "log_ret"
    return r

def ewma_variance_proxy(r: pd.Series, dt: float, lam: float = 0.94, eps: float = 1e-12) -> pd.Series:
    """
    EWMA proxy for instantaneous variance v_t.
    Returns a variance per year (so consistent with dt in years).
    """
    # EWMA of squared returns
    rv = r.pow(2.0)
    vhat = rv.ewm(alpha=(1-lam)).mean() / dt  # annualize by dividing by dt
    vhat = vhat.clip(lower=eps)
    vhat.name = "vhat"
    return vhat

def estimate_variance_dynamics_WLS(vhat: pd.Series, dt: float) -> Tuple[float, float, float]:
    """
    Weighted LS for: Δv_t = κθ dt - κ v_{t-1} dt + ξ sqrt(v_{t-1} dt) * ε_t
    Weights w_t = 1 / (v_{t-1} dt) to account for heteroskedasticity.
    Returns (kappa, theta, xi).
    """
    v_tm1 = vhat.shift(1).dropna()
    v_t   = vhat.loc[v_tm1.index]
    dv    = v_t - v_tm1

    # Design matrix for regression: dv = a*dt + b*(-v_tm1)*dt + noise
    # where a = κθ, b = κ. We'll estimate a and b and then recover κ, θ.
    X = np.column_stack([np.ones_like(v_tm1.values)*dt, -v_tm1.values*dt])  # [dt, -v_tm1*dt]
    y = dv.values

    # Weights ~ 1/Var(noise) ≈ 1 / (ξ^2 v_{t-1} dt). ξ unknown; use w ∝ 1/(v_tm1*dt)
    w = 1.0 / np.maximum(v_tm1.values*dt, 1e-18)
    W = np.sqrt(w)

    Xw = X * W[:, None]
    yw = y * W

    # Solve weighted least squares
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    a_hat, b_hat = beta  # a_hat ≈ κθ, b_hat ≈ κ

    kappa = float(b_hat)
    theta = float(a_hat / max(kappa, 1e-12))

    # Estimate xi from residuals: resid = dv - (a_hat dt + b_hat (-v_tm1) dt)
    resid = y - (X @ beta)
    # Var(resid | v_tm1) ≈ ξ^2 v_{t-1} dt  => ξ^2 ≈ mean( resid^2 / (v_{t-1} dt) )
    xi_sq = np.mean((resid**2) / np.maximum(v_tm1.values*dt, 1e-18))
    xi = float(np.sqrt(max(xi_sq, 0.0)))

    return kappa, theta, xi

def estimate_mu_and_rho(r: pd.Series, vhat: pd.Series, dt: float, kappa: float, theta: float, xi: float) -> Tuple[float, float]:
    """
    From return equation and variance equation residuals:
      r_t = (μ - 0.5 v_{t-1}) dt + sqrt(v_{t-1} dt) * ε1_t
      Δv_t = κ(θ - v_{t-1}) dt + ξ sqrt(v_{t-1} dt) * ε2_t
    Estimate μ via OLS on r_t, then compute ε1, ε2 and set ρ = corr(ε1, ε2).
    """
    idx = (vhat.index.intersection(r.index)).intersection(vhat.index[1:])  # align & ensure t and t-1 exist
    r = r.loc[idx]
    v_tm1 = vhat.shift(1).loc[idx]
    v_t = vhat.loc[idx]
    dv = v_t - v_tm1

    # OLS for mu: r_t + 0.5 v_{t-1} dt = μ dt + sqrt(v_{t-1} dt)*ε1_t
    y = r.values + 0.5 * v_tm1.values * dt
    X = np.ones((len(y), 1)) * dt
    mu_hat = float(np.linalg.lstsq(X, y, rcond=None)[0][0])

    # Standardized shocks
    denom = np.sqrt(np.maximum(v_tm1.values*dt, 1e-18))
    eps1 = (r.values - (mu_hat - 0.5*v_tm1.values)*dt) / denom
    eps2 = (dv.values - kappa*(theta - v_tm1.values)*dt) / (np.maximum(xi,1e-18)*denom)

    rho_hat = float(np.corrcoef(eps1, eps2)[0,1])
    rho_hat = np.clip(rho_hat, -0.999, 0.999)
    return mu_hat, rho_hat

def negative_quasi_loglike(params, r: np.ndarray, vhat_tm1: np.ndarray, dt: float):
    """
    Simple quasi-likelihood using proxy vhat_{t-1}:
      r_t | v_{t-1} ~ N((μ - 0.5 v_{t-1}) dt, v_{t-1} dt)
    and
      Δv_t | v_{t-1} ~ N(κ(θ - v_{t-1}) dt, ξ^2 v_{t-1} dt)
    with corr(ε1,ε2)=ρ (ignored in this simple separable QMLE for robustness).
    """
    mu, kappa, theta, xi = params
    # Return part
    mean_r = (mu - 0.5*vhat_tm1) * dt
    var_r  = np.maximum(vhat_tm1*dt, 1e-18)
    ll_r = -0.5*(np.log(2*np.pi*var_r) + (r - mean_r)**2/var_r)

    # (Optional) variance part could be added if you pass dv and include xi,kappa,theta
    return -np.sum(ll_r)  # minimize negative log-like

def estimate_heston_from_prices(prices: pd.Series, dt: float = 1/252, lam_ewma: float = 0.94) -> HestonParams:
    r = log_returns(prices)
    vhat = ewma_variance_proxy(r, dt, lam=lam_ewma)
    v0 = float(vhat.iloc[0])

    # Step 1: WLS for (kappa, theta, xi)
    kappa, theta, xi = estimate_variance_dynamics_WLS(vhat, dt)

    # Step 2: μ and ρ
    mu, rho = estimate_mu_and_rho(r, vhat, dt, kappa, theta, xi)

    # (Optional) Step 3: refine μ, κ, θ, ξ by QMLE on return eq only (robust & simple)
    idx = r.index.intersection(vhat.index)[1:]
    r_use = r.loc[idx].values
    vhat_tm1 = vhat.shift(1).loc[idx].values
    x0 = np.array([mu, kappa, theta, xi])
    bounds = [(-5, 5), (1e-6, 50.0), (1e-8, 5.0), (1e-6, 5.0)]
    res = minimize(negative_quasi_loglike, x0,
                   args=(r_use, vhat_tm1, dt),
                   method="L-BFGS-B", bounds=bounds)

    mu2, kappa2, theta2, xi2 = res.x
    # Keep rho from step 2 (you can re-estimate using updated params if desired)
    return HestonParams(mu=mu2, kappa=kappa2, theta=theta2, xi=xi2, rho=rho, v0=v0)