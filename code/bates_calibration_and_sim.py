#Bates model: Heston with Poisson jumps

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
import src.bates_model_parameters as bmod
# from src.bates_model_parameters import BatesParams

# ----------------------------
# Config
# ----------------------------
symbol = 'BTCUSD'
timeframe = '1m'
parquet_path = f'data/spot/ohlcv_{symbol}_{timeframe}.parquet'

# Test day to compare
test_start = pd.Timestamp('2025-07-02', tz='UTC')
test_end   = pd.Timestamp('2025-07-13', tz='UTC')


# --- Heston-vs-Jumps weighting knobs ---
jump_lookback_days = 60          # fit jumps on a short, recent window
jump_ewma_half_life_days = 7    # faster variance proxy for jump MLE (7–15)
LAM_CAP_PER_DAY = 0.001            # hard cap on jump intensity per day
S_CAP = 0.001                     # cap on log-jump std
SHRINK_JUMPS_IN_SIM = 0.025       # OPTIONAL: scale jump sizes in simulation (0.7–0.9)


# Training window before test_start
train_lookback_days = 365

# EWMA half-life (days) for minute bars (15–60 typical)
ewma_half_life_days = 10




# 1-minute step in "years" (crypto 24/7)
DT_Y = 1.0 / (365 * 24 * 60)

# Poisson truncation for likelihood (minute λ·dt is tiny; 5–7 is usually fine)
N_MAX = 7

RNG_SEED = np.random.randint(1, 1000000)#12345

# ----------------------------
# Helpers
# ----------------------------
def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
    elif df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    return df.sort_index()

def ewma_variance_proxy(r: pd.Series, dt: float, half_life_days: int) -> pd.Series:
    """EWMA proxy for instantaneous variance v_t (annualized)."""
    steps_per_day = 24 * 60
    hl_steps = half_life_days * steps_per_day
    lam = float(np.exp(-np.log(2) / hl_steps))        # smoothing lambda
    alpha = 1.0 - lam                                 # pandas ewm alpha
    vhat = (r**2).ewm(alpha=alpha).mean() / dt       # annualize by /dt
    return vhat.clip(lower=1e-12).rename("vhat")



# ----------------------------
# Load & prep data
# ----------------------------
df = pd.read_parquet(parquet_path)
df = ensure_utc_index(df)
df['r'] = np.log(df['close']).diff()
df = df.dropna(subset=['r'])

train_start = test_start - pd.Timedelta(days=train_lookback_days)
train_df = df.loc[(df.index >= train_start) & (df.index < test_start)].copy()
test_df  = df.loc[(df.index >= test_start) & (df.index < test_end)].copy()

if len(train_df) < 2000:
    print(f"[WARN] Training window is short ({len(train_df)} mins). Consider increasing train_lookback_days.")

# EWMA proxy for v_t on TRAIN
vhat_train = ewma_variance_proxy(train_df['r'], DT_Y, ewma_half_life_days)

# ----------------------------
# Step 1: Heston (variance dynamics) via your estimator
# ----------------------------
heston = hm.estimate_heston_from_prices(
    train_df['close'],
    dt=DT_Y,
    lam_ewma=float(np.exp(-np.log(2) / (ewma_half_life_days * 24 * 60)))
)
print("---- Heston (variance) params ----")
print(f"kappa={heston.kappa:.6f}  theta={heston.theta:.6f}  xi={heston.xi:.6f}  rho={heston.rho:.6f}  v0={heston.v0:.6f}")
feller = 2*heston.kappa*heston.theta
print(f"Feller 2κθ vs ξ²: {feller:.6f} vs {heston.xi**2:.6f}  => "
      f"{'OK' if feller >= heston.xi**2 else 'VIOLATED'}")

# ----------------------------
# Step 2: Bates jump MLE on a RECENT window (regime-matched)
# ----------------------------
jump_train_df = df.loc[
    (df.index >= test_start - pd.Timedelta(days=jump_lookback_days)) &
    (df.index <  test_start)
].copy()

# EWMA proxy for variance for the jump likelihood (shorter half-life)
steps_per_day = 24 * 60
hl_steps_jump = jump_ewma_half_life_days * steps_per_day
lam_jump = float(np.exp(-np.log(2) / hl_steps_jump))
alpha_jump = 1.0 - lam_jump
vhat_jump = (jump_train_df['r']**2).ewm(alpha=alpha_jump).mean() / DT_Y
vhat_jump = vhat_jump.clip(lower=1e-12)

# FIT with caps: limit lambda and s so diffusion carries more load
lam_upper = LAM_CAP_PER_DAY * 365.0
s_upper   = S_CAP

mu_b, lam_b, m_b, s_b = bmod.fit_bates_jumps_mle(
    jump_train_df['r'],
    vhat_jump,
    DT_Y,
    n_max=N_MAX,
    # if your function accepts bounds kwargs; if not, see "Penalty wrapper" below
    bounds_mu=(-5.0, 5.0),
    bounds_lambda=(1e-8, lam_upper),
    bounds_m=(-1.0, 1.0),
    bounds_s=(1e-6, s_upper),
)

print("---- Bates jump params (recent window, capped) ----")
print(f"mu={mu_b:.6f}  lambda={lam_b:.6f}  m={m_b:.6f}  s={s_b:.6f}")
print(f"λ·dt per minute = {lam_b*DT_Y:.3e}   (~{lam_b/365:.3f} jumps/day)")

# Package params (no name shadowing)
b_params = bmod.BatesParams(
    kappa=heston.kappa, theta=heston.theta, xi=heston.xi, rho=heston.rho, v0=heston.v0,
    mu=mu_b, lam=lam_b, m=m_b, s=s_b
)

# ----------------------------
# Simulate Bates over TEST window (1-minute)
# ----------------------------
def simulate_bates(b_params, test_df):
    n_steps = len(test_df)
    S0 = float(test_df['close'].iloc[0])
    RNG_SEED_2 = np.random.randint(1, 1000000)#12345
    rng = np.random.default_rng(RNG_SEED_2)

    # Correlated Brownian increments
    Z1 = rng.standard_normal(n_steps)
    Zp = rng.standard_normal(n_steps)
    Z2 = b_params.rho * Z1 + np.sqrt(1 - b_params.rho**2) * Zp

    S = np.empty(n_steps + 1); S[0] = S0

    v0_seed = float(vhat_jump.iloc[-1])
    v = np.empty(n_steps + 1); v[0] = v0_seed  # seed variance at boundary
    log_ret_sim = np.empty(n_steps)

    for t in range(n_steps):
        v_pos  = max(v[t], 0.0)
        # Heston step (full truncation Euler)
        v_next = v[t] + b_params.kappa*(b_params.theta - v_pos)*DT_Y + b_params.xi*np.sqrt(v_pos)*np.sqrt(DT_Y)*Z2[t]
        v[t+1] = max(v_next, 1e-12)

        # Jumps for this minute
        Nt = rng.poisson(b_params.lam * DT_Y)
        if Nt > 0:
            jump_sum = rng.normal(loc=Nt*b_params.m, scale=np.sqrt(Nt)*b_params.s)
            if SHRINK_JUMPS_IN_SIM is not None:
                jump_sum *= SHRINK_JUMPS_IN_SIM
        else:
            jump_sum = 0.0

        # Bates log-return increment
        k = np.exp(b_params.m + 0.5*b_params.s*b_params.s) - 1.0
        incr = (b_params.mu - b_params.lam*k - 0.5*v_pos) * DT_Y + np.sqrt(v_pos)*np.sqrt(DT_Y)*Z1[t] + jump_sum

        S[t+1] = S[t] * np.exp(incr)
        log_ret_sim[t] = incr
    return log_ret_sim



# ----------------------------
# Compare distributions (same bins, density)
# ----------------------------
real_ret = test_df['r'].values


plt.figure()
iterations = 25
for i in range(iterations):
    log_ret_sim = simulate_bates(b_params, test_df)
    edges = np.histogram_bin_edges(np.concatenate([real_ret, log_ret_sim]), bins=100)

    plt.hist(log_ret_sim, bins=edges, density=True, alpha=0.35)


plt.hist(real_ret,    bins=edges, fill=False, color='black', density=True, alpha=1, label=f'{symbol} real (1m)')
m = float(np.mean(real_ret)); s = float(np.std(real_ret, ddof=1))
x = np.linspace(m - 4*s, m + 4*s, 200)
plt.plot(x, norm.pdf(x, loc=m, scale=s), label='Normal fit (test window)')

plt.title(f"1-Minute Log-Returns: Real vs Bates (train {train_start.date()} → {test_start.date()})")
plt.xlabel("Log-return (per minute)")
plt.ylabel("Density")
plt.legend()
plt.show()

# Q–Q diagnostics
# plt.figure()
# probplot(real_ret, dist="norm", plot=plt)
# plt.title(f"Q–Q Plot: {symbol} 1m Log-Returns vs Normal (TEST)")
# plt.show()

# plt.figure()
# probplot(log_ret_sim, dist="norm", plot=plt)
# plt.title("Q–Q Plot: Bates 1m Log-Returns vs Normal (SIM)")
# plt.show()

# Optional: tail metrics
def tail_stats(x):
    m, s = np.mean(x), np.std(x, ddof=1)
    return {
        "skew": pd.Series(x).skew(),
        "ex_kurt": pd.Series(x).kurt(),
        "P(|r|>2σ)": float(np.mean(np.abs(x-m) > 2*s)),
        "P(|r|>3σ)": float(np.mean(np.abs(x-m) > 3*s)),
    }
print("Real tails:", tail_stats(real_ret))
print("Bates tails:", tail_stats(log_ret_sim))
