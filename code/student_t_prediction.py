"""
Student-t calibration (real-world P) and simulation for 1-minute BTC log-returns.

Discrete-time model on Δt:
    r_t = alpha Δt + sigma √Δt * T_nu,
where T_nu ~ Student-t(df=nu) i.i.d., with Var[T_nu] = nu/(nu-2) for nu>2.

We estimate (alpha, sigma, nu) by MLE on training log-returns and then
simulate on a test window to compare empirical vs model distributions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

import src.student_t_model_parameters as stmp

# ----------------------------
# Config
# ----------------------------
symbol = 'BTCUSD'
timeframe = '1m'
parquet_path = f'data/spot/ohlcv_{symbol}_{timeframe}.parquet'

test_start = pd.Timestamp('2025-07-01', tz='UTC')
test_end   = pd.Timestamp('2025-07-02', tz='UTC')

train_lookback_days = 90       # history before test_start to fit parameters
DT_Y = 1.0 / (365 * 24 * 60)    # 1-minute step in "years"
RNG_SEED = np.random.randint(1, 1000000)#20240911

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
S0      = float(test_df['close'].iloc[0]) if len(test_df) else float(train_df['close'].iloc[-1])

# ----------------------------
# Fit Student-t by MLE
# ----------------------------
params = stmp.fit_student_t_mle(r_train, DT_Y)

print("--------------------------------")
print("Student-t Parameters (P) estimated on training window:")
print(f"alpha: {params.alpha:.6f}  [per year]")
print(f"sigma: {params.sigma:.6f}  [per sqrt(year)]")
print(f"nu:    {params.nu:.4f}   [degrees of freedom]")
print("--------------------------------")

# ----------------------------
# Simulation on the TEST window
# ----------------------------
n_steps = len(test_df)
if n_steps == 0:
    print("[WARN] Empty test window — adjust test_start/test_end.")
    n_steps = 1

rng = np.random.default_rng(RNG_SEED)

# Draw standard Student-t variates and scale
T = rng.standard_t(df=params.nu, size=n_steps)
log_ret_sim = params.alpha * DT_Y + params.sigma * np.sqrt(DT_Y) * T

# Build a simulated price path aligned to test window
S_sim = np.empty(n_steps + 1)
S_sim[0] = S0
S_sim[1:] = S0 * np.exp(np.cumsum(log_ret_sim))

# ----------------------------
# Compare distributions (histograms)
# ----------------------------
if len(r_test) > 0:
    edges = np.histogram_bin_edges(np.concatenate([r_test, log_ret_sim]), bins=100)
else:
    edges = 100

plt.figure()
if len(r_test) > 0:
    plt.hist(r_test, bins=edges, color='black', fill=False, density=True, alpha=1, label=f'{symbol} real ({timeframe})')
plt.hist(log_ret_sim, bins=edges,  density=True, alpha=0.35, label='Student-t sim (1m)')

# Overlay fitted Student-t density around the test mean
if len(r_test) > 0:
    m = float(np.mean(r_test))
    s = float(np.std(r_test, ddof=1))
    x = np.linspace(m - 6*s, m + 6*s, 400)
else:
    m = float(np.mean(log_ret_sim))
    s = float(np.std(log_ret_sim, ddof=1))
    x = np.linspace(m - 6*s, m + 6*s, 400)

loc = params.alpha * DT_Y
scale = params.sigma * np.sqrt(DT_Y)
plt.plot(x, student_t.pdf((x - loc) / max(scale,1e-18), df=params.nu) / max(scale,1e-18),
         label='Fitted Student-t PDF')

plt.title(f"{timeframe} Log-Returns: Real vs Student-t (train {train_start.date()} → {test_start.date()})")
plt.xlabel("Log-return (per minute)")
plt.ylabel("Density")
plt.legend()
plt.show()

