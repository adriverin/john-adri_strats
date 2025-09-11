import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import probplot
import src.heston_model_parameters as hm  # <-- the estimator we discussed

# ----------------------------
# Config
# ----------------------------
symbol = 'BTCUSD'
timeframe = '1m'
parquet_path = f'data/spot/ohlcv_{symbol}_{timeframe}.parquet'

# Define the test window you want to analyze/compare
test_start = pd.Timestamp('2025-07-01', tz='UTC')  # set tz=None unless your index has tz
test_end   = pd.Timestamp('2025-07-02', tz='UTC')

# How much past data to use to estimate parameters (expanding window works too)
train_lookback_days = 365  # use 6 months of history prior to test_start
# EWMA half-life (in days) for minute bars: tune 15–60 days typically
ewma_half_life_days = 30

# Crypto runs 24/7
DT_Y = 1.0 / (365 * 24 * 60)   # 1 minute in "years"

# ----------------------------
# Load & prep data
# ----------------------------
df = pd.read_parquet(parquet_path)
# Ensure datetime index and sorted
if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
df = df.sort_index()

# Compute returns once
df['r'] = np.log(df['close']).diff()
df = df.dropna(subset=['r'])
mean = df['r'].mean()
std = df['r'].std()

# Split train/test using more historical data for estimation
train_start = test_start - pd.Timedelta(days=train_lookback_days)
train_df = df.loc[(df.index >= train_start) & (df.index < test_start)].copy()
test_df  = df.loc[(df.index >= test_start) & (df.index < test_end)].copy()

if len(train_df) < 2000:
    print(f"[WARN] Training window is short ({len(train_df)} mins). Consider increasing train_lookback_days.")

# ----------------------------
# EWMA lambda for minute bars
# ----------------------------
steps_per_day = 24 * 60
half_life_steps = ewma_half_life_days * steps_per_day
lam_minute = float(np.exp(-np.log(2) / half_life_steps))  # λ = exp(-ln2 / HL_steps)

# ----------------------------
# Estimate Heston parameters on TRAINING PRICES
# ----------------------------
params = hm.estimate_heston_from_prices(
    train_df['close'],
    dt=DT_Y,
    lam_ewma=lam_minute
)

print("--------------------------------")
print("Heston Parameters (P) estimated on training window:")
print(f"mu:    {params.mu:.6f}")
print(f"kappa: {params.kappa:.6f}")
print(f"theta: {params.theta:.6f}")
print(f"xi:    {params.xi:.6f}")
print(f"rho:   {params.rho:.6f}")
print(f"v0:    {params.v0:.6f}")
# Feller condition diagnostic
feller = 2 * params.kappa * params.theta
print(f"Feller 2κθ vs ξ²: {feller:.6f} vs {params.xi**2:.6f}  => "
      f"{'OK' if feller >= params.xi**2 else 'VIOLATED'}")
print("--------------------------------")

# ----------------------------
# Simulate Heston over TEST horizon (same frequency)
# ----------------------------
def simulate_heston(params, test_df):
    n_steps = len(test_df)
    S0 = float(test_df['close'].iloc[0])
    mu, kappa, theta, xi, rho, v0 = (params.mu, params.kappa, params.theta, params.xi, params.rho, params.v0)
    eps = 1e-12

    
    Z1 = np.random.standard_normal(n_steps)
    Zp = np.random.standard_normal(n_steps)
    # rng = np.random.default_rng(12345)
    # Z1 = rng.standard_normal(n_steps)
    # Zp = rng.standard_normal(n_steps)
    
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Zp

    S = np.empty(n_steps + 1); S[0] = S0
    v = np.empty(n_steps + 1); v[0] = v0
    log_ret_sim = np.empty(n_steps)

    for t in range(n_steps):
        v_pos = max(v[t], 0.0)
        v_next = v[t] + kappa*(theta - v_pos)*DT_Y + xi*np.sqrt(v_pos)*np.sqrt(DT_Y)*Z2[t]
        v[t+1] = max(v_next, eps)

        incr = (mu - 0.5*v_pos)*DT_Y + np.sqrt(v_pos)*np.sqrt(DT_Y)*Z1[t]
        S[t+1] = S[t] * np.exp(incr)
        log_ret_sim[t] = incr
    return log_ret_sim



iterations = 100
real_ret = test_df['r'].values


plt.figure()
for i in range(iterations):
    log_ret_sim = simulate_heston(params, test_df)
    edges = np.histogram_bin_edges(np.concatenate([real_ret, log_ret_sim]), bins=100)

    plt.hist(log_ret_sim, bins=edges, density=True, alpha=0.35)

    


plt.hist(real_ret, bins=edges, fill=False, color='black', density=True, alpha=1, label=f'{symbol} real (1m)')
x = np.linspace(mean - 4*std, mean + 4*std, 100)
y = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
plt.plot(x, y, label='Normal Distribution', color='red', linewidth=0.5, linestyle='--')    
plt.title(f"Minute Log-Returns from {test_start.date()} to {test_end.date()}: Real vs {iterations} Heston simulations (train {train_start.date()} → {test_start.date()})")
plt.xlabel("Log-return (per minute)")
plt.ylabel("Density")
plt.legend()
plt.show()



# ----------------------------
# Compare distributions (same bins, density)
# ----------------------------
# real_ret = test_df['r'].values
# edges = np.histogram_bin_edges(np.concatenate([real_ret, log_ret_sim]), bins=100)

# plt.figure()
# plt.hist(real_ret, bins=edges, density=True, alpha=0.55, label=f'{symbol} real (1m)')
# plt.hist(log_ret_sim, bins=edges, density=True, alpha=0.65, label='Heston sim (1m)')
# x = np.linspace(mean - 4*std, mean + 4*std, 100)
# y = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
# plt.plot(x, y, label='Normal Distribution', color='red')
# plt.title(f"Minute Log-Returns: Real vs Heston (train {train_start.date()} → {test_start.date()})")
# plt.xlabel("Log-return (per minute)")
# plt.ylabel("Density")
# plt.legend()
# plt.show()

# # Q–Q for real vs normal (diagnostic)
# plt.figure()
# probplot(real_ret, dist="norm", plot=plt)
# plt.title(f"Q–Q Plot: {symbol} 1m Log-Returns vs Normal (TEST)")
# plt.show()

# # Q–Q for sim vs normal
# plt.figure()
# probplot(log_ret_sim, dist="norm", plot=plt)
# plt.title("Q–Q Plot: Heston 1m Log-Returns vs Normal (SIM)")
# plt.show()
