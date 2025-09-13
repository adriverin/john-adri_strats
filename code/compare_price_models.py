import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, norminvgauss, wasserstein_distance

# Parameter estimators from src/
import src.heston_model_parameters as hm
import src.merton_jump_model_parameters as mjmp
import src.student_t_model_parameters as stmp
import src.bates_model_parameters as bmod


# ----------------------------
# Utilities
# ----------------------------
def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
    elif df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    return df.sort_index()


def timeframe_to_dt_years(timeframe: str) -> float:
    tf = timeframe.lower().strip()
    if tf.endswith('m'):
        minutes = int(tf[:-1])
        return (minutes / (365.0 * 24 * 60))
    if tf.endswith('h'):
        hours = int(tf[:-1])
        return (hours / (365.0 * 24))
    if tf.endswith('d'):
        days = int(tf[:-1])
        return (days / 365.0)
    raise ValueError(f"Unsupported timeframe '{timeframe}'. Use like '1m', '5m', '1h', '1d'.")


def steps_per_day(timeframe: str) -> float:
    tf = timeframe.lower().strip()
    if tf.endswith('m'):
        minutes = int(tf[:-1])
        return 24 * 60 / minutes
    if tf.endswith('h'):
        hours = int(tf[:-1])
        return 24 / hours
    if tf.endswith('d'):
        days = int(tf[:-1])
        return 1.0 / days
    raise ValueError(f"Unsupported timeframe '{timeframe}'.")


def load_ohlcv(symbol: str, timeframe: str, base_path: str = 'data/spot') -> pd.DataFrame:
    p = f"{base_path}/ohlcv_{symbol}_{timeframe}.parquet"
    df = pd.read_parquet(p)
    df = ensure_utc_index(df)
    if 'close' not in df:
        raise ValueError(f"Parquet file missing 'close' column: {p}")
    df['r'] = np.log(df['close']).diff()
    df = df.dropna(subset=['r'])
    return df


def ewma_variance_proxy(r: pd.Series, dt: float, timeframe: str, half_life_days: int) -> pd.Series:
    spd = steps_per_day(timeframe)
    hl_steps = half_life_days * spd
    lam = float(np.exp(-np.log(2.0) / max(hl_steps, 1e-12)))
    alpha = 1.0 - lam
    vhat = (r**2).ewm(alpha=alpha).mean() / dt
    return vhat.clip(lower=1e-12)


# ----------------------------
# Model Runners (Modular Adapters)
# ----------------------------
def compute_histogram_probs(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(x, bins=edges)
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts, dtype=float)
    return counts.astype(float) / float(total)


def compare_histograms_metrics(real: np.ndarray, sim: np.ndarray, edges: np.ndarray) -> Dict[str, float]:
    """
    Compare real vs simulated distributions via shared-bin histograms.
    Returns a dict with Total Variation, Jensen-Shannon divergence, Hellinger distance,
    and 1D Wasserstein distance computed on bin centers with histogram weights.
    """
    p = compute_histogram_probs(real, edges)
    q = compute_histogram_probs(sim, edges)

    # Total Variation distance: 0.5 * L1
    tv = 0.5 * float(np.sum(np.abs(p - q)))

    # Jensen-Shannon divergence (natural log)
    m = 0.5 * (p + q)
    eps = 1e-18
    def kl_div(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * (np.log(a[mask] + eps) - np.log(b[mask] + eps))))
    js = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)

    # Hellinger distance
    hell = (1.0 / np.sqrt(2.0)) * float(np.linalg.norm(np.sqrt(p) - np.sqrt(q)))

    # Wasserstein distance using bin centers and weights
    centers = 0.5 * (edges[:-1] + edges[1:])
    wd = float(wasserstein_distance(centers, centers, u_weights=p, v_weights=q))

    return dict(tv=tv, js=js, hellinger=hell, wasserstein=wd)

class BaseRunner:
    name: str = "base"

    def fit(self, train_df: pd.DataFrame, timeframe: str, dt: float, cfg: dict) -> object:
        raise NotImplementedError

    def simulate(self, params: object, test_df: pd.DataFrame, timeframe: str, dt: float, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def param_dict(self, params: object) -> Dict[str, float]:
        raise NotImplementedError


class HestonRunner(BaseRunner):
    name = "heston"

    def fit(self, train_df: pd.DataFrame, timeframe: str, dt: float, cfg: dict) -> hm.HestonParams:
        # EWMA lambda for given timeframe
        spd = steps_per_day(timeframe)
        hl_days = int(cfg.get('heston_ewma_half_life_days', 30))
        lam = float(np.exp(-np.log(2.0) / max(hl_days * spd, 1e-12)))
        params = hm.estimate_heston_from_prices(train_df['close'], dt=dt, lam_ewma=lam)
        return params

    def simulate(self, params: hm.HestonParams, test_df: pd.DataFrame, timeframe: str, dt: float, rng: np.random.Generator) -> np.ndarray:
        n_steps = len(test_df)
        if n_steps == 0:
            return np.array([])
        S0 = float(test_df['close'].iloc[0])
        mu, kappa, theta, xi, rho, v0 = (params.mu, params.kappa, params.theta, params.xi, params.rho, params.v0)
        eps = 1e-12

        Z1 = rng.standard_normal(n_steps)
        Zp = rng.standard_normal(n_steps)
        Z2 = rho * Z1 + np.sqrt(max(1 - rho**2, 0.0)) * Zp

        S = np.empty(n_steps + 1); S[0] = S0
        v = np.empty(n_steps + 1); v[0] = v0
        log_ret_sim = np.empty(n_steps)

        for t in range(n_steps):
            v_pos = max(v[t], 0.0)
            v_next = v[t] + kappa*(theta - v_pos)*dt + xi*np.sqrt(v_pos)*np.sqrt(dt)*Z2[t]
            v[t+1] = max(v_next, eps)

            incr = (mu - 0.5*v_pos)*dt + np.sqrt(v_pos)*np.sqrt(dt)*Z1[t]
            S[t+1] = S[t] * np.exp(incr)
            log_ret_sim[t] = incr
        return log_ret_sim

    def param_dict(self, p: hm.HestonParams) -> Dict[str, float]:
        return dict(mu=p.mu, kappa=p.kappa, theta=p.theta, xi=p.xi, rho=p.rho, v0=p.v0,
                    feller_2kth=2*p.kappa*p.theta, xi_sq=p.xi*p.xi)


class MertonRunner(BaseRunner):
    name = "merton"

    def fit(self, train_df: pd.DataFrame, timeframe: str, dt: float, cfg: dict) -> mjmp.MertonParams:
        n_max = int(cfg.get('merton_n_max', 7))
        r_train = train_df['r'].values
        params = mjmp.fit_merton_mle(r_train, dt, n_max=n_max)
        return params

    def simulate(self, params: mjmp.MertonParams, test_df: pd.DataFrame, timeframe: str, dt: float, rng: np.random.Generator) -> np.ndarray:
        n_steps = len(test_df)
        if n_steps == 0:
            return np.array([])
        Z = rng.standard_normal(n_steps)
        N = rng.poisson(params.lam * dt, size=n_steps)
        jump_component = np.zeros(n_steps)
        nonzero = N > 0
        if np.any(nonzero):
            jump_component[nonzero] = rng.normal(loc=N[nonzero]*params.m,
                                                 scale=np.sqrt(N[nonzero])*params.s)
        log_ret_sim = params.alpha * dt + params.sigma * np.sqrt(dt) * Z + jump_component
        return log_ret_sim

    def param_dict(self, p: mjmp.MertonParams) -> Dict[str, float]:
        return dict(alpha=p.alpha, sigma=p.sigma, lam=p.lam, m=p.m, s=p.s, mu=p.mu,
                    lam_per_day=p.lam/365.0)


class StudentTRunner(BaseRunner):
    name = "student_t"

    def fit(self, train_df: pd.DataFrame, timeframe: str, dt: float, cfg: dict) -> stmp.StudentTParams:
        r_train = train_df['r'].values
        params = stmp.fit_student_t_mle(r_train, dt)
        return params

    def simulate(self, params: stmp.StudentTParams, test_df: pd.DataFrame, timeframe: str, dt: float, rng: np.random.Generator) -> np.ndarray:
        n_steps = len(test_df)
        if n_steps == 0:
            return np.array([])
        T = rng.standard_t(df=params.nu, size=n_steps)
        log_ret_sim = params.alpha * dt + params.sigma * np.sqrt(dt) * T
        return log_ret_sim

    def param_dict(self, p: stmp.StudentTParams) -> Dict[str, float]:
        return dict(alpha=p.alpha, sigma=p.sigma, nu=p.nu)


class BatesRunner(BaseRunner):
    name = "bates"

    def fit(self, train_df: pd.DataFrame, timeframe: str, dt: float, cfg: dict) -> bmod.BatesParams:
        # First, estimate Heston on train prices
        spd = steps_per_day(timeframe)
        hl_days = int(cfg.get('heston_ewma_half_life_days', 30))
        lam_heston = float(np.exp(-np.log(2.0) / max(hl_days * spd, 1e-12)))
        heston = hm.estimate_heston_from_prices(train_df['close'], dt=dt, lam_ewma=lam_heston)

        # Jump MLE on a recent subset (optionally shorter lookback); here reuse full train for simplicity
        jump_hl_days = int(cfg.get('bates_jump_ewma_half_life_days', 7))
        vhat = ewma_variance_proxy(train_df['r'], dt, timeframe, half_life_days=jump_hl_days)

        n_max = int(cfg.get('bates_n_max', 7))
        lam_cap_per_day = float(cfg.get('bates_lam_cap_per_day', 0.02))
        s_cap = float(cfg.get('bates_s_cap', 0.02))
        lam_upper = lam_cap_per_day * 365.0

        mu_b, lam_b, m_b, s_b = bmod.fit_bates_jumps_mle(
            train_df['r'], vhat, dt, n_max=n_max,
            bounds_mu=(-5.0, 5.0),
            bounds_lambda=(1e-8, lam_upper),
            bounds_m=(-1.0, 1.0),
            bounds_s=(1e-6, s_cap),
        )

        return bmod.BatesParams(
            kappa=heston.kappa, theta=heston.theta, xi=heston.xi, rho=heston.rho, v0=heston.v0,
            mu=mu_b, lam=lam_b, m=m_b, s=s_b,
        )

    def simulate(self, p: bmod.BatesParams, test_df: pd.DataFrame, timeframe: str, dt: float, rng: np.random.Generator) -> np.ndarray:
        n_steps = len(test_df)
        if n_steps == 0:
            return np.array([])
        # Correlated Brownian increments
        Z1 = rng.standard_normal(n_steps)
        Zp = rng.standard_normal(n_steps)
        Z2 = p.rho * Z1 + np.sqrt(max(1 - p.rho**2, 0.0)) * Zp

        # Use last observed variance proxy as v0 seed if available
        # Here use p.v0 as provided; keep strictly positive
        v = np.empty(n_steps + 1); v[0] = max(p.v0, 1e-12)
        log_ret_sim = np.empty(n_steps)

        for t in range(n_steps):
            v_pos = max(v[t], 0.0)
            v_next = v[t] + p.kappa*(p.theta - v_pos)*dt + p.xi*np.sqrt(v_pos)*np.sqrt(dt)*Z2[t]
            v[t+1] = max(v_next, 1e-12)

            # Jumps for this step
            Nt = rng.poisson(p.lam * dt)
            if Nt > 0:
                jump_sum = rng.normal(loc=Nt*p.m, scale=np.sqrt(Nt)*p.s)
            else:
                jump_sum = 0.0

            k = np.exp(p.m + 0.5*p.s*p.s) - 1.0
            incr = (p.mu - p.lam*k - 0.5*v_pos) * dt + np.sqrt(v_pos)*np.sqrt(dt)*Z1[t] + jump_sum
            log_ret_sim[t] = incr
        return log_ret_sim

    def param_dict(self, p: bmod.BatesParams) -> Dict[str, float]:
        return dict(kappa=p.kappa, theta=p.theta, xi=p.xi, rho=p.rho, v0=p.v0,
                    mu=p.mu, lam=p.lam, m=p.m, s=p.s, lam_per_day=p.lam/365.0)


class GBMRunner(BaseRunner):
    name = "gbm"

    def fit(self, train_df: pd.DataFrame, timeframe: str, dt: float, cfg: dict) -> Dict[str, float]:
        r = train_df['r'].values
        mean_r = float(np.mean(r))
        std_r = float(np.std(r, ddof=1))
        # Log-return model: r_t = alpha*dt + sigma*sqrt(dt) Z
        alpha = mean_r / max(dt, 1e-18)
        sigma = std_r / max(np.sqrt(dt), 1e-18)
        mu = alpha + 0.5 * sigma * sigma
        return dict(alpha=alpha, sigma=sigma, mu=mu)

    def simulate(self, params: Dict[str, float], test_df: pd.DataFrame, timeframe: str, dt: float, rng: np.random.Generator) -> np.ndarray:
        n_steps = len(test_df)
        if n_steps == 0:
            return np.array([])
        Z = rng.standard_normal(n_steps)
        return params['alpha'] * dt + params['sigma'] * np.sqrt(dt) * Z

    def param_dict(self, p: Dict[str, float]) -> Dict[str, float]:
        return dict(alpha=p['alpha'], sigma=p['sigma'], mu=p['mu'])


class GARCHRunner(BaseRunner):
    name = "garch"

    @staticmethod
    def _softplus(x: float) -> float:
        return float(np.log1p(np.exp(-abs(x))) + max(x, 0.0))

    def _unpack_params(self, theta: np.ndarray):
        # theta = [m_raw, w_raw, a_raw, b_raw, h0_raw]
        m = theta[0]
        w = self._softplus(theta[1]) + 1e-12
        a_raw = self._softplus(theta[2])
        b_raw = self._softplus(theta[3])
        s = 1.0 + a_raw + b_raw
        a = a_raw / s
        b = b_raw / s
        # ensure a+b < 1
        a = float(min(max(a, 1e-8), 0.999 - 1e-6))
        b = float(min(max(b, 1e-8), 0.999 - a - 1e-6))
        h0 = self._softplus(theta[4]) + 1e-12
        return m, w, a, b, h0

    def _neg_loglike(self, theta: np.ndarray, r: np.ndarray) -> float:
        m, w, a, b, h0 = self._unpack_params(theta)
        T = r.shape[0]
        e = np.empty(T)
        h = np.empty(T)
        ll = 0.0
        h_prev = h0
        for t in range(T):
            # variance update uses previous residual
            if t == 0:
                e_prev_sq = 0.0
            else:
                e_prev_sq = e[t-1]**2
            h_t = w + a * e_prev_sq + b * h_prev
            h_t = max(h_t, 1e-18)
            h[t] = h_t
            e[t] = r[t] - m
            ll += 0.5 * (np.log(2*np.pi) + np.log(h_t) + (e[t]**2)/h_t)
            h_prev = h_t
        return float(ll)

    def fit(self, train_df: pd.DataFrame, timeframe: str, dt: float, cfg: dict) -> Dict[str, float]:
        from scipy.optimize import minimize
        r = train_df['r'].values.astype(float)
        # initial guesses
        m0 = float(np.mean(r))
        var0 = float(np.var(r, ddof=1))
        a0 = 0.05
        b0 = 0.90
        w0 = max(var0 * (1 - a0 - b0), 1e-8)
        # inverse softplus to get raw initializations, approximate with log(exp(x)-1)
        def inv_softplus(y: float) -> float:
            return float(np.log(np.expm1(max(y, 1e-12))))
        x0 = np.array([
            m0,
            inv_softplus(w0),
            inv_softplus(a0),
            inv_softplus(b0),
            inv_softplus(var0),
        ], dtype=float)
        res = minimize(self._neg_loglike, x0, args=(r,), method='L-BFGS-B')
        if not res.success:
            print("[WARN] GARCH MLE did not fully converge:", res.message)
        m, w, a, b, h0 = self._unpack_params(res.x)
        # convert mean to annualized alpha for display
        alpha = m / max(dt, 1e-18)
        uncond = (w/(1-a-b) if (a+b) < 0.999 else np.nan)
        return dict(m=m, w=w, a=a, b=b, h0=h0, alpha=alpha, a_plus_b=a+b, uncond_var=uncond)

    def simulate(self, p: Dict[str, float], test_df: pd.DataFrame, timeframe: str, dt: float, rng: np.random.Generator) -> np.ndarray:
        n_steps = len(test_df)
        if n_steps == 0:
            return np.array([])
        m, w, a, b = p['m'], p['w'], p['a'], p['b']
        h_prev = float(p.get('uncond_var', 0.0))
        if not np.isfinite(h_prev) or h_prev <= 0:
            h_prev = max(p['h0'], 1e-12)
        z = rng.standard_normal(n_steps)
        r_sim = np.empty(n_steps)
        e_prev = 0.0
        for t in range(n_steps):
            h_t = max(w + a * (e_prev**2) + b * h_prev, 1e-18)
            e_t = np.sqrt(h_t) * z[t]
            r_sim[t] = m + e_t
            e_prev = e_t
            h_prev = h_t
        return r_sim

    def param_dict(self, p: Dict[str, float]) -> Dict[str, float]:
        return p


class NormalIIDRunner(BaseRunner):
    name = "normal"

    def fit(self, train_df: pd.DataFrame, timeframe: str, dt: float, cfg: dict) -> Dict[str, float]:
        r = train_df['r'].values
        mean_r = float(np.mean(r))
        std_r = float(np.std(r, ddof=1))
        return dict(mean_r=mean_r, std_r=std_r)

    def simulate(self, params: Dict[str, float], test_df: pd.DataFrame, timeframe: str, dt: float, rng: np.random.Generator) -> np.ndarray:
        n_steps = len(test_df)
        if n_steps == 0:
            return np.array([])
        return rng.normal(loc=params['mean_r'], scale=max(params['std_r'], 1e-18), size=n_steps)

    def param_dict(self, p: Dict[str, float]) -> Dict[str, float]:
        return p


class NIGRunner(BaseRunner):
    name = "nig"

    def fit(self, train_df: pd.DataFrame, timeframe: str, dt: float, cfg: dict) -> Dict[str, float]:
        r = train_df['r'].values
        # Fit NIG parameters (a>0, |b|<a) with loc and scale
        a, b, loc, scale = norminvgauss.fit(r)
        return dict(a=a, b=b, loc=loc, scale=scale)

    def simulate(self, p: Dict[str, float], test_df: pd.DataFrame, timeframe: str, dt: float, rng: np.random.Generator) -> np.ndarray:
        n_steps = len(test_df)
        if n_steps == 0:
            return np.array([])
        # Use scipy random_state to ensure reproducibility
        return norminvgauss.rvs(p['a'], p['b'], loc=p['loc'], scale=max(p['scale'], 1e-18), size=n_steps, random_state=rng)

    def param_dict(self, p: Dict[str, float]) -> Dict[str, float]:
        return p


# ----------------------------
# Registry
# ----------------------------
MODEL_REGISTRY: Dict[str, BaseRunner] = {
    HestonRunner.name: HestonRunner(),
    MertonRunner.name: MertonRunner(),
    StudentTRunner.name: StudentTRunner(),
    BatesRunner.name: BatesRunner(),
    GBMRunner.name: GBMRunner(),
    GARCHRunner.name: GARCHRunner(),
    NormalIIDRunner.name: NormalIIDRunner(),
    NIGRunner.name: NIGRunner(),
}


# ----------------------------
# Orchestrator
# ----------------------------
def run_models(
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_lookback_days: int,
    models: List[str],
    seed: int,
    cfg: dict,
    save_path: Optional[str] = None,
    do_plot: bool = True,
    print_metrics: bool = True,
) -> Dict[str, Dict[str, float]]:
    # Load data once
    df = load_ohlcv(symbol, timeframe)
    dt = timeframe_to_dt_years(timeframe)

    # Split train/test windows
    train_start = start - pd.Timedelta(days=train_lookback_days)
    train_df = df.loc[(df.index >= train_start) & (df.index < start)].copy()
    test_df = df.loc[(df.index >= start) & (df.index < end)].copy()

    if train_df.empty:
        raise ValueError("Training window empty. Adjust --train-lookback-days or --start/--end.")
    if test_df.empty:
        print("[WARN] Test window empty; charts will reflect no test data.")

    # Prepare data and RNG
    real_ret = test_df['r'].values
    rng = np.random.default_rng(seed)

    # Fit and simulate all selected models first
    results = []  # list of dicts: {'name':, 'params':, 'p_dict':, 'sim': np.ndarray}
    for model_name in models:
        runner = MODEL_REGISTRY[model_name]
        try:
            params = runner.fit(train_df, timeframe, dt, cfg)
            p_dict = runner.param_dict(params)
            sim_ret = runner.simulate(params, test_df, timeframe, dt, rng)
        except Exception as e:
            print(f"[ERROR] {model_name}: {e}")
            continue

        # Print parameters (only when plotting is enabled)
        if do_plot:
            print("--------------------------------")
            print(f"{model_name.title()} parameters (train window):")
            for k, v in p_dict.items():
                if isinstance(v, (float, int)) and np.isfinite(v):
                    print(f"  {k}: {float(v):.6f}")
                else:
                    print(f"  {k}: {v}")
            if model_name == 'heston':
                feller = p_dict.get('feller_2kth', np.nan)
                xi_sq = p_dict.get('xi_sq', np.nan)
                status = 'OK' if feller >= xi_sq else 'VIOLATED'
                print(f"  Feller 2κθ vs ξ²: {feller:.6f} vs {xi_sq:.6f} => {status}")
            print("--------------------------------")

        results.append(dict(name=model_name, params=params, p_dict=p_dict, sim=sim_ret))

    # Determine global x-range across real and all sim returns
    mins = []
    maxs = []
    if real_ret.size:
        mins.append(np.min(real_ret))
        maxs.append(np.max(real_ret))
    for res in results:
        if res['sim'].size:
            mins.append(np.min(res['sim']))
            maxs.append(np.max(res['sim']))
    if mins and maxs:
        xmin = float(min(mins))
        xmax = float(max(maxs))
        pad = 0.02 * (xmax - xmin if xmax > xmin else 1.0)
        xlim = (xmin - pad, xmax + pad)
    else:
        xlim = None

    # Common histogram edges across all subplots for consistent bin width
    bins_count = int(cfg.get('bins_count', 100))
    if xlim is not None:
        edges_global = np.linspace(xlim[0], xlim[1], bins_count + 1)
    else:
        edges_global = None
    if print_metrics:
        print(f"[INFO] Using global histogram bins: {bins_count}")

    # Compute and print histogram-based divergences for each model
    if edges_global is None:
        edges_for_metrics = np.histogram_bin_edges(real_ret, bins=bins_count) if real_ret.size else np.linspace(-1, 1, bins_count+1)
    else:
        edges_for_metrics = edges_global

    if print_metrics:
        print("==== Histogram Divergences vs Real (shared bins) ====")
        print("Lower is better. TV∈[0,1], JS∈[0,ln2], Hellinger∈[0,1], Wasserstein in return units.")
        print("All metrics computed from probability histograms using shared bin edges.")
        for res in results:
            sim_ret = res['sim']
            if not (real_ret.size and sim_ret.size):
                print(f"{res['name'].title()}: [no data to compare]")
                continue
            metrics = compare_histograms_metrics(real_ret, sim_ret, edges_for_metrics)
            print(f"{res['name'].title()}: TV={metrics['tv']:.4f}  JS={metrics['js']:.4f}  Hellinger={metrics['hellinger']:.4f}  Wasserstein={metrics['wasserstein']:.6f}")
    # Prepare return structure
    metrics_by_model: Dict[str, Dict[str, float]] = {}
    for res in results:
        sim_ret = res['sim']
        if (real_ret.size and sim_ret.size):
            metrics_by_model[res['name']] = compare_histograms_metrics(real_ret, sim_ret, edges_for_metrics)

    if not do_plot:
        return metrics_by_model

    # Plot grid
    n = len(results)
    if n == 0:
        print("No models produced results.")
        return metrics_by_model
    # dynamic grid: up to 3 columns
    cols = 1 if n == 1 else (2 if n <= 4 else 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0*cols, 3.8*rows), squeeze=False, sharex=True)

    for idx, res in enumerate(results):
        ax = axes[divmod(idx, cols)[0]][divmod(idx, cols)[1]]
        sim_ret = res['sim']
        # Use a single common set of edges across all subplots
        edges = edges_global if edges_global is not None else bins_count

        if real_ret.size:
            ax.hist(real_ret, bins=edges, density=True, alpha=1, fill=False, color='black', label=f'Real {symbol} {timeframe} log-returns', histtype='stepfilled')
        if sim_ret.size:
            ax.hist(sim_ret, bins=edges, density=True, alpha=0.25, label=f"{res['name']} sim")

        if real_ret.size:
            m = float(np.mean(real_ret)); s = float(np.std(real_ret, ddof=1))
            if s > 0:
                # Normal fit to test window (reference only)
                x = np.linspace(m - 4*s, m + 4*s, 200)
                ax.plot(x, norm.pdf(x, loc=m, scale=s), linestyle='--', linewidth=0.9, color='red', label='Normal (test fit)')
                # If this is the Normal model subplot, also overlay the model-implied normal
                if res['name'] == 'normal':
                    mean_r = res['p_dict'].get('mean_r', m)
                    std_r = max(res['p_dict'].get('std_r', s), 1e-18)
                    ax.plot(x, norm.pdf(x, loc=mean_r, scale=std_r), linestyle=':', linewidth=1.0, color='blue', label='Normal (model)')

        ax.set_title(res['name'].title())
        ax.set_xlabel('Log-return')
        ax.set_ylabel('Density')
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.legend(fontsize=8)

    fig.suptitle(f"{symbol} {timeframe}: Real vs Model Simulated Log-Returns\nTrain {train_start.date()} → {start.date()} | Test {start.date()} → {end.date()}", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()
    return metrics_by_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare price prediction models: Heston, Merton, Student-t, Bates, GBM, GARCH.")
    parser.add_argument('--symbol', type=str, default='BTCUSD', help='Symbol, e.g., BTCUSD')
    parser.add_argument('--timeframe', type=str, default='1m', help="Bar timeframe like '1m', '5m', '1h'")
    parser.add_argument('--start', type=str, required=True, help="Test window start (YYYY-MM-DD)")
    parser.add_argument('--end', type=str, required=False, help="Test window end (YYYY-MM-DD)")
    parser.add_argument('--train-lookback-days', type=int, default=365, help='Days before start used for training')
    parser.add_argument('--models', type=str, default='all', help="Comma-separated model names to run. Default: all. Choices: heston,merton,student_t,bates,gbm,garch,normal,nig")
    parser.add_argument('--seed', type=int, default=12345, help='RNG seed')
    parser.add_argument('--save', type=str, default='', help='Optional path to save the figure instead of showing')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting; only print metrics')
    parser.add_argument('--bins', type=int, default=100, help='Number of histogram bins (global across all subplots)')
    parser.add_argument('--days-count', type=int, default=1, help='Number of daily windows to compare starting at --start')
    parser.add_argument('--weekly', action='store_true', help='Jump 7 days between windows (same weekday across weeks)')

    # Optional knobs (can be extended without changing code structure)
    parser.add_argument('--heston-ewma-half-life-days', type=int, default=30)
    parser.add_argument('--bates-jump-ewma-half-life-days', type=int, default=7)
    parser.add_argument('--bates-lam-cap-per-day', type=float, default=0.02)
    parser.add_argument('--bates-s-cap', type=float, default=0.02)
    parser.add_argument('--merton-n-max', type=int, default=7)
    parser.add_argument('--bates-n-max', type=int, default=7)

    return parser.parse_args()


def main():
    args = parse_args()

    start = pd.Timestamp(args.start, tz='UTC')
    end = pd.Timestamp(args.end, tz='UTC')
    if end <= start:
        raise ValueError('--end must be after --start')

    # Select models
    if args.models.strip().lower() == 'all':
        models = list(MODEL_REGISTRY.keys())
    else:
        requested = [m.strip().lower() for m in args.models.split(',') if m.strip()]
        unknown = [m for m in requested if m not in MODEL_REGISTRY]
        if unknown:
            raise ValueError(f"Unknown models: {unknown}. Available: {list(MODEL_REGISTRY.keys())}")
        models = requested

    # Config bag to allow easy future extension without changing signatures
    cfg = dict(
        heston_ewma_half_life_days=int(args.heston_ewma_half_life_days),
        bates_jump_ewma_half_life_days=int(args.bates_jump_ewma_half_life_days),
        bates_lam_cap_per_day=float(args.bates_lam_cap_per_day),
        bates_s_cap=float(args.bates_s_cap),
        merton_n_max=int(args.merton_n_max),
        bates_n_max=int(args.bates_n_max),
        bins_count=int(args.bins),
    )

    save_path = args.save if args.save else None

    if int(args.days_count) <= 1:
        run_models(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start=start,
            end=end,
            train_lookback_days=int(args.train_lookback_days),
            models=models,
            seed=int(args.seed),
            cfg=cfg,
            save_path=save_path,
            do_plot=(not args.no_plot),
            print_metrics=True,
        )
    else:
        step_days = 7 if bool(args.weekly) else 1
        avg_metrics: Dict[str, Dict[str, float]] = {}
        counts: Dict[str, int] = {}
        n_windows = 0
        for i in range(int(args.days_count)):
            day_start = start + pd.Timedelta(days=i * step_days)
            day_end = day_start + pd.Timedelta(days=1)
            if save_path:
                sp = save_path
                dot = sp.rfind('.')
                if dot > 0:
                    save_i = f"{sp[:dot]}_d{i+1}{sp[dot:]}"
                else:
                    save_i = f"{sp}_d{i+1}.png"
            else:
                save_i = None

            m = run_models(
                symbol=args.symbol,
                timeframe=args.timeframe,
                start=day_start,
                end=day_end,
                train_lookback_days=int(args.train_lookback_days),
                models=models,
                seed=int(args.seed),
                cfg=cfg,
                save_path=save_i,
                do_plot=(not args.no_plot) and (i == 0),
                print_metrics=False,
            )
            # accumulate
            if m:
                n_windows += 1
                for model_name, vals in m.items():
                    if model_name not in avg_metrics:
                        avg_metrics[model_name] = {k: 0.0 for k in vals.keys()}
                        counts[model_name] = 0
                    for k, v in vals.items():
                        avg_metrics[model_name][k] += float(v)
                    counts[model_name] += 1

        if n_windows > 0:
            print("==== Average Histogram Divergences over", n_windows, "window(s) ====")
            print("Lower is better. TV∈[0,1], JS∈[0,ln2], Hellinger∈[0,1], Wasserstein in return units.")
            for model_name, sums in avg_metrics.items():
                c = counts.get(model_name, 0)
                if c > 0:
                    avg = {k: (s / c) for k, s in sums.items()}
                    print(f"{model_name.title()}: TV={avg['tv']:.4f}  JS={avg['js']:.4f}  Hellinger={avg['hellinger']:.4f}  Wasserstein={avg['wasserstein']:.6f}  (n={c})")
        else:
            print("[WARN] No windows produced metrics. Check data availability and dates.")


if __name__ == '__main__':
    main()
