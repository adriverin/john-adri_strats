import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot
import src.heston_model_parameters as hm  # your module with estimate_heston_from_prices

symbol = 'BTCUSD'
timeframe = '1m'
start_date = '2024-03-03'
end_date   = '2024-03-06'

df = pd.read_parquet(f'data/spot/ohlcv_{symbol}_{timeframe}.parquet')
df = df[(df.index >= start_date) & (df.index < end_date)].copy()



# 1) Minute log-returns (no forward shift)
df['r'] = np.log(df['close']).diff()
df = df.dropna()

mean = df['r'].mean()
std = df['r'].std()

# 2) Minute dt in years (crypto 24/7)
DT_Y = 1.0 / (365 * 24 * 60)

# 3) EWMA lambda for minute bars via half-life (e.g., 30 days)
half_life_days = 30
steps_per_day = 24 * 60
half_life_steps = half_life_days * steps_per_day
lam_minute = float(np.exp(-np.log(2) / half_life_steps))

# 4) Estimate Heston (real-world P) with correct dt and λ
heston_params = hm.estimate_heston_from_prices(
    df['close'],
    dt=DT_Y,
    lam_ewma=lam_minute
)

print("--------------------------------")
print("Heston Parameters (P):")
print(f"mu:    {heston_params.mu:.6f}")
print(f"kappa: {heston_params.kappa:.6f}")
print(f"theta: {heston_params.theta:.6f}")
print(f"xi:    {heston_params.xi:.6f}")
print(f"rho:   {heston_params.rho:.6f}")
print(f"v0:    {heston_params.v0:.6f}")
print("--------------------------------")

# 5) Simulate Heston for the SAME horizon/steps/frequency
n_steps = len(df)                     # one increment per minute
S0 = float(df['close'].iloc[0])
mu, kappa, theta, xi, rho, v0 = (heston_params.mu, heston_params.kappa,
                                 heston_params.theta, heston_params.xi,
                                 heston_params.rho, heston_params.v0)
eps = 1e-12

rng = np.random.default_rng(12345)
Z1 = rng.standard_normal(n_steps)
Zp = rng.standard_normal(n_steps)
Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Zp

S = np.empty(n_steps + 1); S[0] = S0
v = np.empty(n_steps + 1); v[0] = v0
log_ret = np.empty(n_steps)

for t in range(n_steps):
    v_pos = max(v[t], 0.0)
    v_next = v[t] + kappa*(theta - v_pos)*DT_Y + xi*np.sqrt(v_pos)*np.sqrt(DT_Y)*Z2[t]
    v[t+1] = max(v_next, eps)

    incr = (mu - 0.5*v_pos)*DT_Y + np.sqrt(v_pos)*np.sqrt(DT_Y)*Z1[t]
    S[t+1] = S[t] * np.exp(incr)
    log_ret[t] = incr  # minute log-returns in years time scale

# 6) Common bin edges + density for fair comparison
edges = np.histogram_bin_edges(
    np.concatenate([log_ret, df['r'].values]),
    bins=200
)

plt.figure()
plt.hist(log_ret, bins=edges, density=True, alpha=0.7, label='Heston (sim, 1m)')
plt.hist(df['r'], bins=edges, density=True, alpha=0.5, label=f'{symbol} (real, 1m)')
x = np.linspace(mean - 4*std, mean + 4*std, 100)
y = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
plt.plot(x, y, label='Normal Distribution', color='red')
plt.title("Histogram of Minute Log-Returns (Heston vs Real)")
plt.xlabel("Log-return (per minute)")
plt.ylabel("Density")
plt.legend()
plt.show()

# Optional: Q–Q plots
plt.figure()
probplot(df['r'], dist="norm", plot=plt)
plt.title(f"Q–Q Plot: {symbol} 1m Log-Returns vs Normal")
plt.show()

plt.figure()
probplot(log_ret, dist="norm", plot=plt)
plt.title("Q–Q Plot: Heston 1m Log-Returns vs Normal")
plt.show()
