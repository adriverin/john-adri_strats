import numpy as np
from dataclasses import dataclass
from typing import Tuple
from scipy.optimize import minimize
from scipy.special import gammaln


@dataclass
class StudentTParams:
    """
    Parameters for i.i.d. Student-t log-returns model at step size dt (in years):

        r_t = alpha * dt + sigma * sqrt(dt) * T_nu

    where T_nu is a standard Student-t random variable with df=nu.
    Note: Var[T_nu] = nu/(nu-2) for nu>2 => Var[r_t] = sigma^2 * dt * nu/(nu-2).
    """
    alpha: float  # drift (per year) in the log-return equation
    sigma: float  # scale (per sqrt(year)) multiplying the t-innovations
    nu: float     # degrees of freedom (nu > 2)


def _student_t_logpdf(x: np.ndarray, loc: float, scale: float, nu: float) -> np.ndarray:
    """
    Log-density of the Student-t with df=nu, location=loc, scale=scale.
    x, loc, scale are in the same units; scale>0, nu>0.
    """
    z = (x - loc) / scale
    # log c = log Gamma((nu+1)/2) - log Gamma(nu/2) - 0.5*log(nu*pi) - log(scale)
    logc = (
        gammaln((nu + 1.0) * 0.5)
        - gammaln(nu * 0.5)
        - 0.5 * np.log(nu * np.pi)
        - np.log(scale)
    )
    # -(nu+1)/2 * log(1 + z^2/nu)
    logk = -0.5 * (nu + 1.0) * np.log1p((z * z) / nu)
    return logc + logk


def _neg_loglike_student_t(params_vec: np.ndarray, r: np.ndarray, dt: float) -> float:
    """
    Negative log-likelihood for i.i.d. Student-t increments.
    params_vec = [alpha, sigma, nu] with sigma>0 and nu>2.
    """
    alpha, sigma, nu = params_vec
    if sigma <= 0.0 or nu <= 2.0:
        return np.inf

    loc = alpha * dt
    scale = sigma * np.sqrt(dt)
    # Guard for tiny scale
    scale = float(max(scale, 1e-18))

    ll = _student_t_logpdf(r, loc=loc, scale=scale, nu=nu)
    nll = -float(np.sum(ll))
    return nll


def fit_student_t_mle(r: np.ndarray, dt: float,
                      *,
                      bounds_alpha: Tuple[float, float] = (-5.0, 5.0),
                      bounds_sigma: Tuple[float, float] = (1e-8, 5.0),
                      bounds_nu: Tuple[float, float] = (2.001, 200.0),
                      x0: np.ndarray | None = None,
                      opt_options: dict | None = None) -> StudentTParams:
    """
    Fit Student-t parameters by MLE on an array of log-returns r, for step size dt (years).

    Returns StudentTParams(alpha, sigma, nu) where the model is:
        r_t = alpha*dt + sigma*sqrt(dt) * T_nu
    """
    r = np.asarray(r, dtype=float)

    # Initial guesses
    if x0 is None:
        mean_r = float(np.mean(r))
        std_r = float(np.std(r, ddof=1))
        nu0 = 6.0  # moderately heavy tails
        # std_r^2 ≈ sigma^2 * dt * nu/(nu-2) => sigma ≈ std_r/sqrt(dt) * sqrt((nu-2)/nu)
        sigma0 = (std_r / max(np.sqrt(dt), 1e-12)) * np.sqrt(max(nu0 - 2.0, 1e-6) / nu0)
        alpha0 = mean_r / max(dt, 1e-12)
        x0 = np.array([alpha0, max(sigma0, 1e-6), nu0], dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float)

    bounds = [bounds_alpha, bounds_sigma, bounds_nu]
    if opt_options is None:
        opt_options = dict(maxiter=600, ftol=1e-9)

    res = minimize(_neg_loglike_student_t, x0,
                   args=(r, dt), method="L-BFGS-B",
                   bounds=bounds, options=opt_options)
    if not res.success:
        print("[WARN] Student-t MLE did not fully converge:", res.message)

    alpha, sigma, nu = res.x
    return StudentTParams(alpha=float(alpha), sigma=float(sigma), nu=float(nu))

