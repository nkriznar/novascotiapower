# Nova Scotia Electricity Demand Forecasting Pipeline

This repository contains a robust data preparation and feature engineering pipeline designed for time-series forecasting of electricity consumption.

## Project Overview

The core of this project is a modular Python pipeline that transforms raw historical data into an enriched dataset ready for machine learning models. It handles everything from data cleaning to advanced feature extraction and chronological dataset splitting.

## Pipeline Architecture

The pipeline, implemented in `energy_pipeline.py`, follows a four-stage process:

### 1. Data Ingestion & Cleaning
- **Chronological Integrity**: The dataset is sorted by timestamp to ensure temporal ordering.
- **Robust Missing Value Handling**:
    - **Forward-fill (`ffill`)**: Carries the last known observation forward.
    - **Linear Interpolation**: Smoothly fills gaps in continuous variables (consumption and weather).
    - **Backward-fill (`bfill`)**: Extracts initial values for any leading gaps.

### 2. Feature Engineering
The pipeline generates **11 new features** to capture temporal patterns:

| Category | Features | Description |
|---|---|---|
| **Lag Features** | `lag_1`, `lag_24`, `lag_168` | Captures consumption from 1 hour, 1 day, and 1 week ago. |
| **Rolling Stats** | `rolling_mean_24h`, `rolling_mean_168h` | 24-hour and 7-day moving averages to capture trends. |
| **Calendar** | `hour`, `day_of_week`, `month` | Explicitly encodes cyclic time patterns. |
| **Contextual** | `is_weekend`, `is_holiday` | Boolean flags; holidays are specific to **Nova Scotia, Canada**. |

### 3. Data Splitting
- **Time-Based Split**: Unlike random sampling, this uses a strict chronological split (80% Train, 20% Test) to prevent "data leakage" (training on future data to predict the past).
- **Cold Start Handling**: Automatically removes the initial rows (168 hours) that contain NaNs due to the lag/rolling window requirements.

### 4. Output Generation
Saves two ready-to-use CSV files (excluded from the repository to maintain performance):
- `train_data_engineered.csv`
- `test_data_engineered.csv`

---

## Getting Started

### Prerequisites
- Python 3.8+
- [requirements.txt](requirements.txt) dependencies: `pandas`, `numpy`, `holidays`

### Installation
```bash
pip install -r requirements.txt
```

### Usage
1. Ensure your raw data file (`nsp_electricity_dataset.csv`) is in the project directory.
2. Run the pipeline:
```bash
python energy_pipeline.py
```

## Repository Structure
- `energy_pipeline.py`: Main processing script.
- `requirements.txt`: Python package dependencies.
- `.gitignore`: Configured to exclude large data files and Python artifacts.
