"""
Energy Data Preparation & Feature Engineering Pipeline (v2)
============================================================
A production-grade, config-driven pipeline for cleaning, enriching, and
splitting historical electricity consumption data for time-series forecasting.

Key upgrades over v1
--------------------
* All parameters loaded from ``config.yaml`` — zero hardcoded values.
* Cyclical sine / cosine encoding for hour, day-of-week, and month.
* Heating & Cooling Degree Days (HDD / CDD) from a configurable baseline.
* Holiday proximity features (days until next / since last holiday).

Author : Nikola Kriznar (MSc CDA – Data Mining)
Instructor: Pranay 
Date   : 2026-03-12
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import holidays
import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Config = Dict[str, Any]


# ===========================================================================
# 0. Configuration Loader
# ===========================================================================
def load_config(config_path: str | Path = "config.yaml") -> Config:
    """Read and return the YAML configuration file as a dictionary.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML config file (default: ``config.yaml`` in the
        script's directory).

    Returns
    -------
    Config
        Parsed configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg: Config = yaml.safe_load(fh)

    return cfg


# ===========================================================================
# 1. Data Ingestion & Cleaning
# ===========================================================================
def load_and_clean(cfg: Config) -> pd.DataFrame:
    """Load the CSV, parse timestamps, set the index, and fill missing values.

    The strategy preserves chronological integrity:
      1. Forward-fill (``ffill``) carries the last valid observation forward.
      2. Linear interpolation fills any remaining interior gaps.
      3. Backward-fill (``bfill``) handles leading NaNs.

    Parameters
    ----------
    cfg : Config
        Pipeline configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with a ``DatetimeIndex`` named *timestamp*.
    """
    filepath: Path = Path(cfg["paths"]["input_csv"])
    if not filepath.exists():
        print(f"[ERROR] Input file not found: {filepath}")
        sys.exit(1)

    df: pd.DataFrame = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # Columns to clean: target + weather columns from config
    target: str = cfg["columns"]["target"]
    weather_cols: List[str] = cfg["columns"]["weather"]
    fill_cols: List[str] = [target] + [
        c for c in weather_cols if c != target
    ]

    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
            df[col] = df[col].interpolate(method="linear")
            df[col] = df[col].bfill()

    return df


# ===========================================================================
# 2. Feature Engineering – Individual Feature Blocks
# ===========================================================================

# ---- 2a. Lag Features ----------------------------------------------------
def add_lag_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Create historical lag columns for the target variable.

    The lag intervals are read from ``cfg["features"]["lag_hours"]``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the target column.
    cfg : Config
        Pipeline configuration dictionary.

    Returns
    -------
    pd.DataFrame
        DataFrame with new ``lag_<n>`` columns appended.
    """
    target: str = cfg["columns"]["target"]
    lag_hours: List[int] = cfg["features"]["lag_hours"]

    for lag in lag_hours:
        df[f"lag_{lag}"] = df[target].shift(lag)

    return df


# ---- 2b. Rolling Statistics ---------------------------------------------
def add_rolling_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Create rolling-mean columns for the target variable.

    Window sizes are read from ``cfg["features"]["rolling_windows"]``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the target column.
    cfg : Config
        Pipeline configuration dictionary.

    Returns
    -------
    pd.DataFrame
        DataFrame with new ``rolling_mean_<w>h`` columns appended.
    """
    target: str = cfg["columns"]["target"]
    windows: List[int] = cfg["features"]["rolling_windows"]

    for w in windows:
        df[f"rolling_mean_{w}h"] = (
            df[target].rolling(window=w, min_periods=1).mean()
        )

    return df


# ---- 2c. Cyclical Time Encoding -----------------------------------------
def add_cyclical_time(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour, day-of-week, and month as sine / cosine pairs.

    Mapping temporal features onto a unit circle preserves their cyclical
    continuity (e.g. hour 23 is close to hour 0).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``DatetimeIndex``.

    Returns
    -------
    pd.DataFrame
        DataFrame with six new columns:
        ``hour_sin``, ``hour_cos``, ``dow_sin``, ``dow_cos``,
        ``month_sin``, ``month_cos``.
    """
    hour: np.ndarray = df.index.hour.to_numpy(dtype=np.float64)
    dow: np.ndarray = df.index.dayofweek.to_numpy(dtype=np.float64)
    month: np.ndarray = df.index.month.to_numpy(dtype=np.float64)

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    return df


# ---- 2d. Degree Days (Domain-Specific) ----------------------------------
def add_degree_days(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute Heating Degree Days (HDD) and Cooling Degree Days (CDD).

    *  HDD = max(0, base − temperature)  →  demand for space heating
    *  CDD = max(0, temperature − base)  →  demand for air-conditioning

    The base comfort threshold is read from
    ``cfg["features"]["base_temperature_c"]`` (default 18 °C).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the temperature column.
    cfg : Config
        Pipeline configuration dictionary.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``heating_degree_days`` and ``cooling_degree_days``.
    """
    temp_col: str = cfg["columns"]["temperature"]
    base: float = cfg["features"]["base_temperature_c"]

    temp: pd.Series = df[temp_col]
    df["heating_degree_days"] = np.maximum(0.0, base - temp)
    df["cooling_degree_days"] = np.maximum(0.0, temp - base)

    return df


# ---- 2e. Holiday Features (Boolean + Proximity) -------------------------
def add_holiday_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add is_weekend, is_holiday, and holiday proximity features.

    Proximity features capture the commercial ramp-down / ramp-up in energy
    consumption that occurs around public holidays.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``DatetimeIndex``.
    cfg : Config
        Pipeline configuration dictionary.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``is_weekend``, ``is_holiday``,
        ``days_until_next_holiday``, and ``days_since_last_holiday``.
    """
    # --- is_weekend -------------------------------------------------------
    df["is_weekend"] = np.where(df.index.dayofweek >= 5, 1, 0)

    # --- Build holiday calendar -------------------------------------------
    country: str = cfg["holidays"]["country"]
    province: str = cfg["holidays"]["province"]

    year_min: int = df.index.min().year   # type: ignore[union-attr]
    year_max: int = df.index.max().year   # type: ignore[union-attr]

    ns_holidays = holidays.country_holidays(
        country=country,
        prov=province,
        years=range(year_min, year_max + 1),
    )

    # Sorted array of holiday dates as datetime64 for vectorised search
    holiday_dates: np.ndarray = np.sort(
        np.array(list(ns_holidays.keys()), dtype="datetime64[D]")
    )

    # --- is_holiday -------------------------------------------------------
    dates_normalised: np.ndarray = df.index.normalize().values.astype(
        "datetime64[D]"
    )
    df["is_holiday"] = np.isin(dates_normalised, holiday_dates).astype(int)

    # --- Holiday proximity ------------------------------------------------
    # Use np.searchsorted for O(log n) proximity computation per row.
    date_vals: np.ndarray = dates_normalised.astype("int64")
    hol_vals: np.ndarray = holiday_dates.astype("int64")

    # days_until_next_holiday
    idx_right: np.ndarray = np.searchsorted(hol_vals, date_vals, side="right")
    idx_right_clipped: np.ndarray = np.clip(idx_right, 0, len(hol_vals) - 1)
    days_until: np.ndarray = (
        (hol_vals[idx_right_clipped] - date_vals)
        // 86_400_000_000_000  # nanoseconds → days
    )
    # If past the last holiday, cap at 0 (no future holiday in range)
    days_until = np.where(idx_right >= len(hol_vals), 0, days_until)
    df["days_until_next_holiday"] = days_until.astype(int)

    # days_since_last_holiday
    idx_left: np.ndarray = np.searchsorted(hol_vals, date_vals, side="left") - 1
    idx_left_clipped: np.ndarray = np.clip(idx_left, 0, len(hol_vals) - 1)
    days_since: np.ndarray = (
        (date_vals - hol_vals[idx_left_clipped])
        // 86_400_000_000_000
    )
    # If before the first holiday, cap at 0
    days_since = np.where(idx_left < 0, 0, days_since)
    df["days_since_last_holiday"] = days_since.astype(int)

    return df


# ===========================================================================
# 2f. Feature Engineering Orchestrator
# ===========================================================================
def engineer_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Apply all feature engineering steps in sequence.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with a ``DatetimeIndex``.
    cfg : Config
        Pipeline configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Fully enriched DataFrame.
    """
    df = df.copy()
    df = add_lag_features(df, cfg)
    df = add_rolling_features(df, cfg)
    df = add_cyclical_time(df)
    df = add_degree_days(df, cfg)
    df = add_holiday_features(df, cfg)
    return df


# ===========================================================================
# 3. Time-Based Train / Test Split
# ===========================================================================
def split_time_series(
    df: pd.DataFrame,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a strict chronological train / test split.

    No random sampling is used; the first ``train_ratio`` fraction of rows
    becomes the training set and the remainder becomes the test set.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame sorted by its DatetimeIndex.
    cfg : Config
        Pipeline configuration dictionary (reads ``split.train_ratio``).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``(train_df, test_df)`` preserving chronological order.
    """
    train_ratio: float = cfg["split"]["train_ratio"]
    split_idx: int = int(len(df) * train_ratio)
    train_df: pd.DataFrame = df.iloc[:split_idx]
    test_df: pd.DataFrame = df.iloc[split_idx:]
    return train_df, test_df


# ===========================================================================
# 4. Main Orchestrator
# ===========================================================================
def main() -> None:
    """Run the full data-preparation pipeline end-to-end.

    Steps
    -----
    1. Load configuration from ``config.yaml``.
    2. Ingest and clean the raw CSV.
    3. Engineer advanced time-series features.
    4. Drop rows with NaN values from lag / rolling warm-up.
    5. Split chronologically and save output CSVs.
    """
    script_dir: Path = Path(__file__).resolve().parent
    cfg: Config = load_config(script_dir / "config.yaml")

    print("=" * 60)
    print("  Energy Data Pipeline v2 – Config-Driven")
    print("=" * 60)

    # Step 1 – Ingest & clean
    print("\n[1/4] Loading and cleaning data …")
    df: pd.DataFrame = load_and_clean(cfg)
    print(f"       Loaded {len(df):,} rows  |  Columns: {len(df.columns)}")

    # Step 2 – Feature engineering
    print("[2/4] Engineering features …")
    df = engineer_features(df, cfg)
    new_features: int = df.shape[1]
    print(f"       Features added → {new_features} total columns")

    # Step 3 – Drop lag / rolling NaN rows
    rows_before: int = len(df)
    df = df.dropna()
    rows_after: int = len(df)
    print(
        f"[3/4] Dropped {rows_before - rows_after} initial NaN rows "
        f"(lag/rolling warm-up)  →  {rows_after:,} rows remaining"
    )

    # Step 4 – Split & save
    train_df, test_df = split_time_series(df, cfg)

    train_path: Path = script_dir / cfg["paths"]["train_output"]
    test_path: Path = script_dir / cfg["paths"]["test_output"]

    train_df.to_csv(train_path)
    test_df.to_csv(test_path)

    print(f"[4/4] Saved outputs:")
    print(f"       • {train_path}  ({len(train_df):,} rows)")
    print(f"       • {test_path}   ({len(test_df):,} rows)")

    # Summary
    print("\n" + "=" * 60)
    print("  Pipeline Complete ✓")
    print("=" * 60)
    print(f"\n  Date range : {df.index.min()} → {df.index.max()}")
    print(f"  Train rows : {len(train_df):,}")
    print(f"  Test rows  : {len(test_df):,}")
    print(f"  Columns    : {list(df.columns)}\n")


if __name__ == "__main__":
    main()
