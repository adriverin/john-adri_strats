import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# mean = 0.1
# std = 1
# n_steps = 100000
# mean_list = []
# random_list = []
# a = 0

# for i in range(n_steps):
#     a = np.random.normal(mean, std, 1)
#     z_value = (a[0] - mean)/std
#     random_list.append(a[0])

#     if z_value > 3.0:
#         mean = mean + np.sqrt(np.log(np.sqrt(z_value)))
#         # a = np.random.normal(mean, std, 1)
#         mean_list.append(mean)
#     elif z_value < -3.0:
#         mean = mean - np.sqrt(np.log(np.sqrt(-1 * z_value)))
#         mean_list.append(mean)

# mean_list = pd.Series(mean_list)
# random_list = pd.Series(random_list)

# mean_of_mean_list = mean_list.mean()
# std_of_mean_list = mean_list.std()

# # plt.plot(mean_list)
# plt.hist(random_list, bins=100, label='Moving Mean Gaussian')
# x = np.linspace(mean - 4*std, mean + 4*std, n_steps)
# plt.plot(x, norm.pdf(x, mean, std), color='red', label='Gaussian Distribution', linewidth=0.5, linestyle='--')
# plt.legend()
# plt.show()


data  = pd.read_parquet('data/spot/ohlcv_BTCUSD_1m.parquet')
data['r'] = np.log(data['close']).diff()
data = data.dropna(subset=['r'])
mean = data['r'].mean()
std = data['r'].std()

plt.hist(data['r'], bins=1000, label=f'BTCUSD 1m {mean:.6f} {std:.6f}')
x = np.linspace(mean - 4*std, mean + 4*std, 10000)
plt.plot(x, norm.pdf(x, mean, std), color='red', label='Gaussian Distribution', linewidth=0.5, linestyle='--')
plt.axvline(mean, color='blue', label='Mean', linewidth=0.5, linestyle='--')
plt.axvline(mean + std, color='green', label='Mean + Std', linewidth=0.5, linestyle='--')
plt.axvline(mean - std, color='green', label='Mean - Std', linewidth=0.5, linestyle='--')
plt.axvline(mean + 2*std, color='yellow', label='Mean + 2*Std', linewidth=0.5, linestyle='--')
plt.axvline(mean - 2*std, color='yellow', label='Mean - 2*Std', linewidth=0.5, linestyle='--')
plt.axvline(mean + 3*std, color='orange', label='Mean + 3*Std', linewidth=0.5, linestyle='--')
plt.axvline(mean - 3*std, color='orange', label='Mean - 3*Std', linewidth=0.5, linestyle='--')

plt.legend()
plt.show()

