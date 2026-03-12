"""
Energy Data Preparation & Feature Engineering Pipeline
=======================================================
A modular pipeline for cleaning, enriching, and splitting historical
electricity consumption data for time-series forecasting.

Author : Pranay (MSc CDA – Data Mining)
Date   : 2026-03-12
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import holidays
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_CSV: str = r"C:\MSc CDA\DataMining\Pranay Assn\nsp_electricity_dataset.csv"

WEATHER_COLS: list[str] = ["temperature", "wind_speed", "humidity"]
TARGET_COL: str = "consumption_kwh"
FILL_COLS: list[str] = [TARGET_COL] + WEATHER_COLS

TRAIN_RATIO: float = 0.80


# ---------------------------------------------------------------------------
# 1. Data Ingestion & Cleaning
# ---------------------------------------------------------------------------
def load_and_clean(filepath: str | Path) -> pd.DataFrame:
    """Load the CSV, parse timestamps, set the index, and fill missing values.

    The strategy preserves chronological integrity:
      1. Forward-fill (``ffill``) carries the last valid observation forward.
      2. Linear interpolation fills any remaining interior gaps.

    Parameters
    ----------
    filepath : str | Path
        Absolute or relative path to the input CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with a ``DatetimeIndex`` named *timestamp*.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    df: pd.DataFrame = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # --- Missing-value handling -------------------------------------------
    # Only touch columns that are expected to be numeric & continuous.
    for col in FILL_COLS:
        if col in df.columns:
            df[col] = df[col].ffill()
            df[col] = df[col].interpolate(method="linear")

    # Back-fill any leading NaNs that ffill could not reach
    for col in FILL_COLS:
        if col in df.columns:
            df[col] = df[col].bfill()

    return df


# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate lag, rolling, calendar, and contextual features.

    Features created
    ----------------
    * **Lag features** – ``lag_1``, ``lag_24``, ``lag_168``
      (1-hour, 24-hour, and 168-hour / 7-day lags of *consumption_kwh*).
    * **Rolling statistics** – ``rolling_mean_24h``, ``rolling_mean_168h``
      (24-hour and 168-hour rolling means of *consumption_kwh*).
    * **Calendar features** – ``hour``, ``day_of_week``, ``month``
      extracted from the DatetimeIndex.
    * **Contextual flags** – ``is_weekend`` (1 for Saturday/Sunday) and
      ``is_holiday`` (1 for Nova-Scotia public holidays using the
      *holidays* package).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with DatetimeIndex and a *consumption_kwh* column.

    Returns
    -------
    pd.DataFrame
        DataFrame enriched with the new feature columns.
    """
    df = df.copy()

    # --- Lag features -----------------------------------------------------
    df["lag_1"] = df[TARGET_COL].shift(1)
    df["lag_24"] = df[TARGET_COL].shift(24)
    df["lag_168"] = df[TARGET_COL].shift(168)

    # --- Rolling statistics -----------------------------------------------
    df["rolling_mean_24h"] = (
        df[TARGET_COL].rolling(window=24, min_periods=1).mean()
    )
    df["rolling_mean_168h"] = (
        df[TARGET_COL].rolling(window=168, min_periods=1).mean()
    )

    # --- Calendar features ------------------------------------------------
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    # --- Contextual flags -------------------------------------------------
    df["is_weekend"] = np.where(df.index.dayofweek >= 5, 1, 0)

    # Build a set of Nova-Scotia holidays spanning the dataset's date range.
    ns_holidays = holidays.Canada(
        prov="NS",
        years=range(
            df.index.min().year,          # type: ignore[union-attr]
            df.index.max().year + 1,      # type: ignore[union-attr]
        ),
    )
    df["is_holiday"] = np.where(
        df.index.normalize().isin(ns_holidays.keys()), 1, 0
    )

    return df


# ---------------------------------------------------------------------------
# 3. Time-Based Train / Test Split
# ---------------------------------------------------------------------------
def split_time_series(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a strict chronological train/test split.

    No random sampling is used; the first *train_ratio* fraction of rows
    becomes the training set and the remainder becomes the test set.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame sorted by its DatetimeIndex.
    train_ratio : float, optional
        Fraction of the timeline allocated to training (default ``0.80``).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``(train_df, test_df)`` — two DataFrames preserving chronological
        order.
    """
    split_idx: int = int(len(df) * train_ratio)
    train_df: pd.DataFrame = df.iloc[:split_idx]
    test_df: pd.DataFrame = df.iloc[split_idx:]
    return train_df, test_df


# ---------------------------------------------------------------------------
# 4. Main Orchestrator
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full data-preparation pipeline end-to-end.

    Steps
    -----
    1. Load and clean the raw CSV.
    2. Engineer time-series features.
    3. Drop rows with NaN values produced by lag / rolling window
       initialisation.
    4. Split chronologically into train (80 %) and test (20 %).
    5. Save ``train_data_engineered.csv`` and ``test_data_engineered.csv``
       to the script's directory.
    """
    print("=" * 60)
    print("  Energy Data Pipeline – Starting")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 – Ingest & clean
    # ------------------------------------------------------------------
    print("\n[1/4] Loading and cleaning data …")
    df: pd.DataFrame = load_and_clean(INPUT_CSV)
    print(f"       Loaded {len(df):,} rows  |  Columns: {list(df.columns)}")

    # ------------------------------------------------------------------
    # Step 2 – Feature engineering
    # ------------------------------------------------------------------
    print("[2/4] Engineering features …")
    df = engineer_features(df)
    print(f"       Features added → {df.shape[1]} total columns")

    # ------------------------------------------------------------------
    # Step 3 – Drop lag / rolling NaN rows
    # ------------------------------------------------------------------
    rows_before: int = len(df)
    df = df.dropna()
    rows_after: int = len(df)
    print(
        f"[3/4] Dropped {rows_before - rows_after} initial NaN rows "
        f"(lag/rolling warm-up)  →  {rows_after:,} rows remaining"
    )

    # ------------------------------------------------------------------
    # Step 4 – Split & save
    # ------------------------------------------------------------------
    train_df, test_df = split_time_series(df)

    output_dir: Path = Path(__file__).resolve().parent
    train_path: Path = output_dir / "train_data_engineered.csv"
    test_path: Path = output_dir / "test_data_engineered.csv"

    train_df.to_csv(train_path)
    test_df.to_csv(test_path)

    print(f"[4/4] Saved outputs:")
    print(f"       • {train_path}  ({len(train_df):,} rows)")
    print(f"       • {test_path}   ({len(test_df):,} rows)")

    # Quick summary
    print("\n" + "=" * 60)
    print("  Pipeline Complete ✓")
    print("=" * 60)
    print(f"\n  Date range : {df.index.min()} → {df.index.max()}")
    print(f"  Train rows : {len(train_df):,}")
    print(f"  Test rows  : {len(test_df):,}")
    print(f"  Columns    : {list(df.columns)}\n")


if __name__ == "__main__":
    main()
