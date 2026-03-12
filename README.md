# Nova Scotia Electricity Demand Forecasting Pipeline

A production-grade, **config-driven** data preparation and feature engineering pipeline for time-series forecasting of electricity consumption.

## Project Overview

The pipeline transforms raw historical electricity data into a machine-learning-ready dataset through four automated stages: cleaning, feature engineering, NaN handling, and chronological splitting.

## Architecture

All tuneable parameters live in **`config.yaml`** — no code changes required to adjust file paths, lag intervals, rolling windows, split ratios, or the degree-day baseline.

```
config.yaml            ← edit parameters here
energy_pipeline.py     ← reads config, runs full pipeline
requirements.txt       ← Python dependencies
```

## Pipeline Stages

### 1. Data Ingestion & Cleaning
- Parses timestamps and sets a `DatetimeIndex`
- Fills missing values: **forward-fill → linear interpolation → backward-fill**
- No rows are dropped during cleaning

### 2. Feature Engineering

| Category | Features | Description |
|---|---|---|
| **Lag** | `lag_1`, `lag_24`, `lag_168` | Consumption from 1 h, 1 day, and 1 week ago |
| **Rolling Stats** | `rolling_mean_24h`, `rolling_mean_168h` | Moving averages to capture trends |
| **Cyclical Time** | `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos` | Sine/cosine encoding preserves temporal continuity (hour 23 ≈ hour 0) |
| **Degree Days** | `heating_degree_days`, `cooling_degree_days` | Energy demand proxies relative to an 18 °C comfort baseline |
| **Contextual** | `is_weekend`, `is_holiday` | Boolean flags; holidays specific to **Nova Scotia, Canada** |
| **Holiday Proximity** | `days_until_next_holiday`, `days_since_last_holiday` | Captures commercial ramp-down / ramp-up patterns |

### 3. NaN Warm-Up Removal
Drops the initial rows containing NaNs from lag/rolling window initialisation (168 rows).

### 4. Chronological Train / Test Split
- **80 % Train / 20 % Test** (configurable)
- **No random sampling** — strict time-based split to prevent data leakage

## Output
Two CSV files (excluded from the repo via `.gitignore`):
- `train_data_engineered.csv`
- `test_data_engineered.csv`

---

## Getting Started

### Prerequisites
- Python 3.8+

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
Edit `config.yaml` to set:
- Input/output file paths
- Lag intervals and rolling window sizes
- Train/test split ratio
- Degree-day base temperature
- Holiday country and province

### Usage
1. Place your raw data file (e.g. `nsp_electricity_dataset.csv`) in the project directory.
2. Run the pipeline:
```bash
python energy_pipeline.py
```

## Repository Structure
| File | Purpose |
|---|---|
| `config.yaml` | All tuneable pipeline parameters |
| `energy_pipeline.py` | Main processing script |
| `requirements.txt` | Python package dependencies |
| `.gitignore` | Excludes data files and Python artifacts |
