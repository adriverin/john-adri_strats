"""
Merton jump-diffusion calibration (real-world P) and simulation for 1-minute BTC.

Model in log-returns over Δt:
    r_t = α Δt + σ √Δt Z_t + sum_{i=1}^{N_t} Y_i
where
    N_t ~ Poisson(λ Δt),  Y_i ~ Normal(m, s^2),  Z_t ~ N(0,1) iid.

Link to SDE:
    dS_t/S_t = μ dt + σ dW_t + (J - 1) dN_t, with J = e^Y.
    α = μ - 0.5 σ^2 - λ k,   k = E[J - 1] = exp(m + 0.5 s^2) - 1
=> μ = α + 0.5 σ^2 + λ (exp(m + 0.5 s^2) - 1).

Estimation:
    MLE on {r_t} with discrete-time mixture density:
    f(r) = Σ_{n=0..N_MAX} Pois(n; λΔt) * NormalPDF(r; αΔt + n m, σ^2 Δt + n s^2)
Truncate at small N_MAX (minute Δt ⇒ λΔt is tiny), use log-sum-exp for stability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm, probplot
import src.merton_jump_model_parameters as mjmp
# ----------------------------
# Config
# ----------------------------
symbol = 'BTCUSD'
timeframe = '1m'
parquet_path = f'data/spot/ohlcv_{symbol}_{timeframe}.parquet'

test_start = pd.Timestamp('2025-07-01', tz='UTC')
test_end   = pd.Timestamp('2025-07-02', tz='UTC')

train_lookback_days = 365       # history before test_start to fit parameters
DT_Y = 1.0 / (365 * 24 * 60)    # 1-minute step in "years"
N_MAX = 25                       # Poisson jumps truncation per step (set 3–7 for speed/accuracy tradeoff)
RNG_SEED = 12345

# ----------------------------
# Data load & prep
# ----------------------------
df = pd.read_parquet(parquet_path)

# Ensure tz-aware UTC index
if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
elif df.index.tz is None:
    df.index = df.index.tz_localize('UTC')
else:
    df.index = df.index.tz_convert('UTC')

df = df.sort_index()
df['r'] = np.log(df['close']).diff()
df = df.dropna(subset=['r'])

train_start = test_start - pd.Timedelta(days=train_lookback_days)
train_df = df.loc[(df.index >= train_start) & (df.index < test_start)].copy()
test_df  = df.loc[(df.index >= test_start) & (df.index < test_end)].copy()

if len(train_df) < 2000:
    print(f"[WARN] Short training window ({len(train_df)} minutes). Consider increasing lookback.")

r_train = train_df['r'].values
r_test  = test_df['r'].values
S0      = float(test_df['close'].iloc[0])


# Fit on training data
params = mjmp.fit_merton_mle(r_train, DT_Y, n_max=N_MAX)

print("--------------------------------")
print("Merton Parameters (P) estimated on training window:")
print(f"alpha: {params.alpha:.6f}  [per year]")
print(f"sigma: {params.sigma:.6f}  [per sqrt(year)]")
print(f"lambda:{params.lam:.6f}  [per year]  (~{params.lam/365:.4f} per day)")
print(f"m:     {params.m:.6f}  [mean log-jump]")
print(f"s:     {params.s:.6f}  [std log-jump]")
print(f"mu:    {params.mu:.6f}  [implied drift of S]")
print("--------------------------------")

# ----------------------------
# Simulation on the TEST window
# ----------------------------
n_steps = len(test_df)
rng = np.random.default_rng(RNG_SEED)

# Simulate log-returns for Merton:
# r = alpha dt + sigma sqrt(dt) Z + sum_{i=1}^N Y_i, N ~ Pois(lam dt), Y ~ N(m, s^2)
Z = rng.standard_normal(n_steps)
N = rng.poisson(params.lam * DT_Y, size=n_steps)

# Sum of N normal(m, s^2) is Normal(N*m, N*s^2). We can draw a single normal per step:
jump_component = np.zeros(n_steps)
nonzero = N > 0
if np.any(nonzero):
    jump_component[nonzero] = rng.normal(loc=N[nonzero]*params.m,
                                         scale=np.sqrt(N[nonzero])*params.s)

log_ret_sim = params.alpha * DT_Y + params.sigma * np.sqrt(DT_Y) * Z + jump_component

# Optional: build a simulated price path at 1-minute frequency
S_sim = np.empty(n_steps + 1); S_sim[0] = S0
S_sim[1:] = S0 * np.exp(np.cumsum(log_ret_sim))

# ----------------------------
# Compare distributions (same bins, density)
# ----------------------------
real_ret = r_test
edges = np.histogram_bin_edges(np.concatenate([real_ret, log_ret_sim]), bins=100)

plt.figure()
plt.hist(real_ret,    bins=edges, density=True, alpha=0.55, label=f'{symbol} real (1m)')
plt.hist(log_ret_sim, bins=edges, density=True, alpha=0.65, label='Merton sim (1m)')

m = float(np.mean(real_ret))
s = float(np.std(real_ret, ddof=1))
x = np.linspace(m - 4*s, m + 4*s, 200)
plt.plot(x, norm.pdf(x, loc=m, scale=s), label='Normal fit (test window)')
plt.title(f"1-Minute Log-Returns: Real vs Merton (train {train_start.date()} → {test_start.date()})")
plt.xlabel("Log-return (per minute)")
plt.ylabel("Density")
plt.legend()
plt.show()

# # Q–Q plots
# plt.figure()
# probplot(real_ret, dist="norm", plot=plt)
# plt.title(f"Q–Q Plot: {symbol} 1m Log-Returns vs Normal (TEST)")
# plt.show()

# plt.figure()
# probplot(log_ret_sim, dist="norm", plot=plt)
# plt.title("Q–Q Plot: Merton 1m Log-Returns vs Normal (SIM)")
# plt.show()
