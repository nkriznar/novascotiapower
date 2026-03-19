# Nova Scotia Electricity Demand Forecasting Pipeline

A production-grade, **config-driven end-to-end time-series analytics pipeline** for electricity consumption in Nova Scotia, including:

- Data preparation
- Feature engineering
- Forecasting (Prophet & ARIMA)
- Anomaly detection
- Visualization
- Feature influence analysis

---

## Project Overview

TThis project transforms raw electricity consumption data into actionable insights through a complete analytics workflow:

1. Clean and prepare raw data
2. Engineer time-series features
3. Forecast electricity demand
4. Detect anomalies from prediction errors
5. Analyze feature influence

## Architecture

All tuneable parameters live in **`config.yaml`** — no code changes required to adjust file paths, lag intervals, rolling windows, split ratios, or the degree-day baseline.

```
config.yaml ← edit parameters here
energy_pipeline.py ← data pipeline
model/ ← forecasting + anomaly detection
heatmap/ ← feature influence analysis
```

## Pipeline Stages

### 1. Data Ingestion & Cleaning

- Parses timestamps and sets a `DatetimeIndex`
- Fills missing values: **forward-fill → linear interpolation → backward-fill**
- No rows are dropped during cleaning

### 2. Feature Engineering

| Category              | Features                                             | Description                                                           |
| --------------------- | ---------------------------------------------------- | --------------------------------------------------------------------- |
| **Lag**               | `lag_1`, `lag_24`, `lag_168`                         | Consumption from 1 h, 1 day, and 1 week ago                           |
| **Rolling Stats**     | `rolling_mean_24h`, `rolling_mean_168h`              | Moving averages to capture trends                                     |
| **Cyclical Time**     | `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`       | Sine/cosine encoding preserves temporal continuity (hour 23 ≈ hour 0) |
| **Degree Days**       | `heating_degree_days`, `cooling_degree_days`         | Energy demand proxies relative to an 18 °C comfort baseline           |
| **Contextual**        | `is_weekend`, `is_holiday`                           | Boolean flags; holidays specific to **Nova Scotia, Canada**           |
| **Holiday Proximity** | `days_until_next_holiday`, `days_since_last_holiday` | Captures commercial ramp-down / ramp-up patterns                      |

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

## Forecasting & Anomaly Detection

### Models Used

#### 1. Prophet (Primary Model)

- Captures:
  - Trend
  - Daily / Weekly / Yearly seasonality
- Robust to missing values and irregularities

**Key Features:**

- Automatic seasonality learning
- Downsampling for large datasets
- Outputs interpretable components

---

#### 2. ARIMA (Benchmark Model)

- Used only for comparison
- Automatically selects best parameters (p, d, q)

---

### Evaluation Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

---

### Anomaly Detection

Anomalies are detected based on **forecast residuals**, not raw data. And normalized before Anomaly Detection.

## residual = actual − predicted

#### Method:

- Isolation Forest

#### Output:

- `1` → anomaly
- `0` → normal

## This captures **unexpected deviations from expected demand**

---

## ⚙️ Model Pipeline

The full workflow is automated in:

- model/model_pipeline.py

### Process:

1. Load train/test data
2. Train Prophet model (per region)
3. Generate forecasts
4. Compute residuals
5. Detect anomalies
6. Evaluate model performance
7. Save outputs and plots

---

### 📁 Outputs

| File                                   | Description        |
| -------------------------------------- | ------------------ |
| `forecast_predictions_all_regions.csv` | All predictions    |
| `forecast_metrics_summary.csv`         | Model comparison   |
| `anomaly_events.csv`                   | Detected anomalies |
| `anomaly_summary_by_region.csv`        | Anomaly statistics |

---

## Visualization

Automatically generated plots:

### 1. Forecast vs Actual

- Daily aggregated trend
- Cleaned using IQR

---

### 2. Pattern Analysis (4-in-1)

- Daily pattern (24h)
- Weekly pattern (7 days)
- Yearly pattern (12 months)
- Long-term trend

Includes automatic turning point detection

---

### 3. Anomaly Plot

- Only anomaly points shown
- Compared against normal baseline

---

## (Optional) Feature Influence (Heatmap Module)

This module answers:

> **What drives electricity consumption?**

---

### Method

- Pearson Correlation (|r|)
- Features ranked by influence strength

---

### Output

feature_influence.png

---

### Run Heatmap

```bash
python heatmap/correlation_heatmap.py

```

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

| File                 | Purpose                                  |
| -------------------- | ---------------------------------------- |
| `config.yaml`        | All tuneable pipeline parameters         |
| `energy_pipeline.py` | Main processing script                   |
| `requirements.txt`   | Python package dependencies              |
| `.gitignore`         | Excludes data files and Python artifacts |
| `model/`             | Forecasting + anomaly detection          |
| `heatmap/`           | Feature influence analysis               |
| `outputs/`           | Predictions, metrics, and plots          |
