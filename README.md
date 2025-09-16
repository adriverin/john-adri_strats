# Market Model Comparison Toolkit

This repo contains a small toolkit to calibrate and compare several price/return models against historical data. It focuses on distributional fit of log-returns over a chosen horizon and frequency, with utilities to simulate and visualize results.

Core features:
- Unified runner to compare multiple models on the same data window.
- Histogram-based divergence metrics: Total Variation, Jensen–Shannon, Hellinger, and 1D Wasserstein.
- Optional plots of real vs. simulated distributions with consistent binning.
- Modular estimators for Heston, Merton (jump–diffusion), Bates (Heston + jumps), Student‑t, plus baselines (GBM, GARCH(1,1), Normal i.i.d., NIG).


## Data Expectations
- Parquet files under `data/spot/` named `ohlcv_{SYMBOL}_{TIMEFRAME}.parquet`.
- Must include at least a `close` column.
- Index should be a `DatetimeIndex` or convertible to UTC; the tools will coerce to tz‑aware UTC and sort.
- Log-returns are computed internally as `r = log(close).diff()` on the chosen timeframe.

Example path: `data/spot/ohlcv_BTCUSD_1m.parquet`


## Installation
Tested with Python 3.10+.

Install the minimal requirements (adjust to your environment):

```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scipy matplotlib pyarrow
```

Notes:
- Reading Parquet requires `pyarrow` or `fastparquet`.
- The NIG model uses `scipy.stats.norminvgauss`.


## Main Script: `code/compare_price_models.py`

Compare multiple models on the same window, compute divergence metrics, and optionally plot distributions.

Usage:
```
python code/compare_price_models.py \
  --symbol BTCUSD \
  --timeframe 1m \
  --start 2025-07-01 \
  --end   2025-07-02 \
  --train-lookback-days 365 \
  --models all \
  --bins 100
```

Key arguments:
- `--symbol` Symbol code (e.g., `BTCUSD`).
- `--timeframe` Bar size like `1m`, `5m`, `1h`, `1d`.
- `--start`, `--end` Test window in `YYYY-MM-DD` (UTC).
- `--train-lookback-days` History before `--start` used for calibration.
- `--models` Comma‑separated list or `all`. Choices: `heston,merton,student_t,bates,gbm,garch,normal,nig`.
- `--seed` RNG seed for simulations.
- `--bins` Histogram bin count (shared across panels/metrics).
- `--save` Save figure to a path (PNG recommended). In multi‑day runs, `_d{N}` is appended per day.
- `--no-plot` Disable plotting and parameter prints; only prints metrics.
- `--days-count` Compare multiple daily windows starting at `--start`.
- `--weekly` When set with `--days-count > 1`, advance windows by 7 days (same weekday) instead of consecutive days.

Model‑specific knobs:
- `--heston-ewma-half-life-days` (default 30)
- `--bates-jump-ewma-half-life-days` (default 7)
- `--bates-lam-cap-per-day` (default 0.02)
- `--bates-s-cap` (default 0.02)
- `--merton-n-max` (default 7)
- `--bates-n-max` (default 7)

Behavior:
- Single day (`--days-count 1` or omitted):
  - Prints model parameters and per‑model histogram metrics.
  - Shows or saves a multi‑panel figure (unless `--no-plot`).
- Multiple days (`--days-count > 1`):
  - Suppresses per‑day metric printing and plots for days after the first.
  - Prints only the averaged histogram metrics across all processed days.
  - If `--save` is provided, only the first day’s plot is generated; files are suffixed `_d1`, `_d2`, ...

Examples:
- Single day, all models, show plot:
  `python code/compare_price_models.py --symbol BTCUSD --timeframe 1m --start 2025-07-01 --end 2025-07-02 --train-lookback-days 365`
- Single day, no plot, metrics only:
  `python code/compare_price_models.py --symbol BTCUSD --timeframe 1m --start 2025-07-01 --end 2025-07-02 --no-plot`
- Three weekly windows, average metrics only, save first day’s figure:
  `python code/compare_price_models.py --symbol BTCUSD --timeframe 1m --start 2025-07-01 --end 2025-07-02 --days-count 3 --weekly --save reports/btc_1m_weekly.png`


## Standalone Calibration/Simulation Scripts

These scripts are self‑contained examples for individual models. They read a single Parquet file, calibrate on a training window, simulate on a test window, and plot comparisons. Edit the config block at the top of each file to change symbol, timeframe, and dates; then run with Python.

- `code/heston_predictor.py`
  - Estimates Heston parameters on training prices using an EWMA variance proxy and WLS/QMLE refinement, then simulates returns over the test window.
  - Plots real vs. simulated histograms and an optional normal reference.
  - Run: `python code/heston_predictor.py`

- `code/merton_prediction.py`
  - Calibrates Merton jump‑diffusion via MLE on log‑returns; simulates returns on the test window.
  - Plots distribution comparison and provides optional Q–Q diagnostics in comments.
  - Run: `python code/merton_prediction.py`

- `code/student_t_prediction.py`
  - Fits a Student‑t innovations model by MLE; simulates aligned returns and overlays the fitted Student‑t PDF.
  - Run: `python code/student_t_prediction.py`

- `code/bates_calibration_and_sim.py`
  - Combines Heston variance dynamics with capped jump MLE (Bates model). Useful for regime‑aware calibration with jump caps.
  - Plots distribution comparison and prints tail probabilities and moments.
  - Run: `python code/bates_calibration_and_sim.py`

- `code/price_prediction_heston_model.py`
  - Minimal Heston example: estimate, simulate, and visualize against real returns on a short window.
  - Run: `python code/price_prediction_heston_model.py`

- `code/basic_mc_sim_DI.ipynb`
  - Exploratory notebook with Monte Carlo experiments. Open in Jupyter to run.

Notes:
- `code/run.sh` appears to be a placeholder and is not a reliable driver; prefer calling the Python scripts directly as above.


## Library Modules (under `code/src/`)
- `heston_model_parameters.py` — Estimation utilities for Heston parameters (EWMA variance proxy, WLS for κ,θ,ξ; QMLE refinement; μ and ρ estimation).
- `merton_jump_model_parameters.py` — Merton MLE with numerically stable log‑sum‑exp mixture; returns `MertonParams`.
- `bates_model_parameters.py` — Bates jump MLE conditional on a variance proxy with parameter bounds; returns `(mu, lam, m, s)` or a `BatesParams` in downstream usage.
- `student_t_model_parameters.py` — Student‑t MLE on returns with bounds and sensible initialization; returns `StudentTParams`.


## Tips
- Ensure the Parquet file’s index is UTC or convertible; scripts will coerce and sort.
- For crypto 24/7 data, time in years uses 365×24×60 for 1‑minute steps.
- Heavy‑tail or jump parameters can be sensitive to window choice; use `--train-lookback-days` and, for Bates, the jump‑specific half‑life and caps to stabilize estimates.


## Troubleshooting
- Missing Parquet or columns: verify `data/spot/ohlcv_{SYMBOL}_{TIMEFRAME}.parquet` exists and includes `close`.
- Parquet engine errors: install `pyarrow` or `fastparquet`.
- Plots don’t show in headless environments: provide `--save` to write PNGs instead of `plt.show()`.
