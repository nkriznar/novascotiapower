

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────────
DARK_BG     = "#0f1117"
PANEL_BG    = "#1e2130"
SPINE_COLOR = "#333344"
TICK_COLOR  = "#cccccc"

# Colour palette per feature group
COLOR_STRONG   = "#e05c5c"   # red    – strong correlation
COLOR_MODERATE = "#f0c040"   # yellow – moderate correlation
COLOR_WEAK     = "#6c7a89"   # grey   – weak correlation
COLOR_LAG      = "#f5a623"   # orange – lag features
COLOR_ROLLING  = "#b8e986"   # green  – rolling average features
COLOR_WEATHER  = "#5bc8f5"   # blue   – weather features

# Thresholds
STRONG_THRESHOLD   = 0.7
MODERATE_THRESHOLD = 0.3

# Columns always excluded before computing correlations
# (non-numeric identifiers / free-text categoricals)
_ALWAYS_DROP = [
    "timestamp", "region", "season",
    "customer_type", "anomaly_type",
    "year", "week",
]


# ──────────────────────────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────────────────────────
def load_config(config_path: Path) -> dict:
    """
    Parse config.yaml and return a settings dict.

    Keys used from config.yaml
    --------------------------
    paths.test_output        CSV to read (engineered test split)
    columns.target           Target column  (e.g. consumption_kwh)
    columns.timestamp        Timestamp column name  → dropped
    columns.region           Region column name     → dropped
    columns.weather          List of weather feature names
    features.lag_hours       Lag hour values  (e.g. [1, 24, 168])
    features.rolling_windows Rolling window sizes (e.g. [24, 168])
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at: {config_path}\n"
            "Run from the repo root or pass --config <path>."
        )

    with open(config_path, "r", encoding="utf-8", errors="replace") as fh:
        raw = yaml.safe_load(fh)

    repo_root = config_path.parent

    # ── paths ──────────────────────────────────────────────────────────────
    csv = Path(raw["paths"]["test_output"])
    if not csv.is_absolute():
        csv = repo_root / csv

    # ── columns ────────────────────────────────────────────────────────────
    timestamp_col = raw["columns"].get("timestamp", "timestamp")
    target_col    = raw["columns"]["target"]
    region_col    = raw["columns"].get("region", "region")
    weather_cols  = raw["columns"].get("weather", [])

    # Columns to drop before correlation (config identifiers + always-drop set)
    drop_cols = list({timestamp_col, region_col} | set(_ALWAYS_DROP))

    # ── feature metadata ───────────────────────────────────────────────────
    lag_hours       = raw["features"].get("lag_hours", [1, 24, 168])
    rolling_windows = raw["features"].get("rolling_windows", [24, 168])

    cfg = {
        "csv_path":        csv,
        "target":          target_col,
        "weather_cols":    weather_cols,
        "drop_cols":       drop_cols,
        "lag_hours":       lag_hours,
        "rolling_windows": rolling_windows,
    }

    print(f"[CONFIG] CSV            : {cfg['csv_path']}")
    print(f"[CONFIG] Target column  : {cfg['target']}")
    print(f"[CONFIG] Weather cols   : {cfg['weather_cols']}")
    print(f"[CONFIG] Lag hours      : {cfg['lag_hours']}")
    print(f"[CONFIG] Rolling windows: {cfg['rolling_windows']}")
    print(f"[CONFIG] Dropped cols   : {cfg['drop_cols']}")

    return cfg


def fallback_config(csv_path: Path, target: str) -> dict:
    """Minimal config when no config.yaml is available."""
    return {
        "csv_path":        csv_path,
        "target":          target,
        "weather_cols":    ["temperature_c", "wind_speed_kmh", "humidity_pct"],
        "drop_cols":       list(_ALWAYS_DROP),
        "lag_hours":       [1, 24, 168],
        "rolling_windows": [24, 168],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────
def load_data(cfg: dict) -> pd.Series:
    """
    Load the CSV, drop non-numeric columns, and return a Series of
    absolute Pearson correlations with the target, sorted descending.
    """
    csv_path = cfg["csv_path"]
    target   = cfg["target"]

    print(f"\n[INFO] Reading : {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Raw shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

    df = df.drop(columns=[c for c in cfg["drop_cols"] if c in df.columns])
    df = df.select_dtypes(include=[np.number])
    print(f"[INFO] Numeric features: {df.shape[1]}")

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in dataset.\n"
            f"Available numeric columns: {df.columns.tolist()}"
        )

    corr = (
        df.corr()[target]
          .drop(target)
          .dropna()                         # remove NaN correlations
          .abs()
          .sort_values(ascending=True)      # ascending → longest bar at top
    )
    return corr


# ──────────────────────────────────────────────────────────────────────────────
# Colour logic
# ──────────────────────────────────────────────────────────────────────────────
def assign_colors(corr: pd.Series, cfg: dict) -> list:
    """
    Return a colour for every feature bar.
    Priority: weather > lag > rolling > strong > moderate > weak
    """
    lag_cols  = {f"lag_{h}"            for h in cfg["lag_hours"]}
    roll_cols = {f"rolling_mean_{w}h"  for w in cfg["rolling_windows"]}
    weather   = set(cfg["weather_cols"])

    def _color(col: str, val: float) -> str:
        if col in weather:    return COLOR_WEATHER
        if col in lag_cols:   return COLOR_LAG
        if col in roll_cols:  return COLOR_ROLLING
        if val >= STRONG_THRESHOLD:   return COLOR_STRONG
        if val >= MODERATE_THRESHOLD: return COLOR_MODERATE
        return COLOR_WEAK

    return [_color(col, val) for col, val in corr.items()]


# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────
def plot_feature_influence(corr: pd.Series, cfg: dict, out_dir: Path) -> None:
    """Build and save the ranked feature influence bar chart."""
    colors = assign_colors(corr, cfg)
    target = cfg["target"]
    n      = len(corr)

    fig, ax = plt.subplots(figsize=(11, max(8, n * 0.38)), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    # ── Bars ─────────────────────────────────────────────────────────────
    bars = ax.barh(corr.index, corr.values,
                   color=colors, edgecolor="none", height=0.72)

    # ── Value labels ──────────────────────────────────────────────────────
    for bar, val in zip(bars, corr.values):
        ax.text(
            val + 0.009,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", ha="left",
            color="white", fontsize=8.5, fontweight="bold",
        )

    # ── Threshold reference lines ─────────────────────────────────────────
    ax.axvline(MODERATE_THRESHOLD, color=COLOR_MODERATE,
               lw=1.2, ls="--", alpha=0.75)
    ax.axvline(STRONG_THRESHOLD,   color=COLOR_STRONG,
               lw=1.2, ls="--", alpha=0.75)
    ax.text(MODERATE_THRESHOLD + 0.005, n * 0.015,
            f"moderate\n({MODERATE_THRESHOLD})",
            color=COLOR_MODERATE, fontsize=7.5, va="bottom")
    ax.text(STRONG_THRESHOLD + 0.005,   n * 0.015,
            f"strong\n({STRONG_THRESHOLD})",
            color=COLOR_STRONG,   fontsize=7.5, va="bottom")

    # ── Axes styling ──────────────────────────────────────────────────────
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Absolute Pearson Correlation  |r|",
                  color="#aaaaaa", fontsize=11)
    ax.tick_params(colors=TICK_COLOR, labelsize=9)
    for spine in ["left", "top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(SPINE_COLOR)

    # ── Legend ────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=COLOR_STRONG,
                       label=f"Strong influence  (|r| ≥ {STRONG_THRESHOLD})"),
        mpatches.Patch(color=COLOR_MODERATE,
                       label=f"Moderate influence  ({MODERATE_THRESHOLD} ≤ |r| < {STRONG_THRESHOLD})"),
        mpatches.Patch(color=COLOR_WEAK,
                       label=f"Weak influence  (|r| < {MODERATE_THRESHOLD})"),
        mpatches.Patch(color=COLOR_LAG,
                       label=f"Lag features  {cfg['lag_hours']}h"),
        mpatches.Patch(color=COLOR_ROLLING,
                       label=f"Rolling avg features  {cfg['rolling_windows']}h"),
        mpatches.Patch(color=COLOR_WEATHER,
                       label=f"Weather features"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower right",
        fontsize=8,
        facecolor=PANEL_BG,
        labelcolor="white",
        edgecolor=SPINE_COLOR,
        framealpha=0.95,
    )

    # ── Titles ────────────────────────────────────────────────────────────
    ax.set_title(
        f"Feature Influence on  '{target}'",
        color="white", fontsize=14, pad=14,
    )
    fig.suptitle(
        "Nova Scotia Power – What Drives Electricity Consumption?",
        color="white", fontsize=16, fontweight="bold", y=0.995,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out_path = out_dir / "feature_influence.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"\n[SAVED] {out_path.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# Console summary
# ──────────────────────────────────────────────────────────────────────────────
def print_summary(corr: pd.Series, cfg: dict) -> None:
    lag_cols  = {f"lag_{h}"           for h in cfg["lag_hours"]}
    roll_cols = {f"rolling_mean_{w}h" for w in cfg["rolling_windows"]}
    weather   = set(cfg["weather_cols"])

    def _group(col):
        if col in weather:   return "weather"
        if col in lag_cols:  return "lag"
        if col in roll_cols: return "rolling"
        return "other"

    print(f"\n{'='*64}")
    print(f"  Feature influence ranking  →  '{cfg['target']}'")
    print(f"{'='*64}")
    print(f"  {'Feature':<32} {'|r|':>7}   Strength     Group")
    print(f"  {'-'*58}")

    for col, val in corr.sort_values(ascending=False).items():
        if val >= STRONG_THRESHOLD:
            strength = "★ Strong  "
        elif val >= MODERATE_THRESHOLD:
            strength = "· Moderate"
        else:
            strength = "  Weak    "
        print(f"  {col:<32} {val:>7.4f}   {strength}  [{_group(col)}]")

    print(f"\n  ★ |r| ≥ {STRONG_THRESHOLD}   "
          f"· {MODERATE_THRESHOLD} ≤ |r| < {STRONG_THRESHOLD}   "
          f"  |r| < {MODERATE_THRESHOLD}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NSP feature influence chart – driven by config.yaml"
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config.yaml  (default: <repo_root>/config.yaml)",
    )
    parser.add_argument(
        "--data", default=None,
        help="Override the CSV path set in config (paths.test_output)",
    )
    parser.add_argument(
        "--target", default=None,
        help="Override the target column set in config (columns.target)",
    )
    return parser.parse_args()


def run_heatmap() -> None:
    args = parse_args()

    # ── Locate repo root and config.yaml ────────────────────────────────────
    repo_root   = Path(__file__).resolve().parent.parent  # heatmap/ → repo root
    config_path = Path(args.config) if args.config else repo_root / "config.yaml"

    # ── Load config (or fall back to defaults) ───────────────────────────────
    if config_path.exists():
        print(f"[INFO] Config : {config_path.resolve()}")
        cfg = load_config(config_path)
    else:
        print(f"[WARN] config.yaml not found at {config_path} – using defaults.")
        cfg = fallback_config(
            csv_path=Path(args.data or "test_data_engineered.csv"),
            target=args.target or "consumption_kwh",
        )

    # ── CLI overrides ─────────────────────────────────────────────────────────
    if args.data:
        cfg["csv_path"] = Path(args.data)
    if args.target:
        cfg["target"] = args.target

    if not cfg["csv_path"].exists():
        print(f"[ERROR] CSV not found: {cfg['csv_path']}", file=sys.stderr)
        sys.exit(1)

    # ── Create timestamped output subfolder inside heatmap/outputs/ ──────────
    out_dir = Path(__file__).parent / "outputs" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output : {out_dir.resolve()}")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    corr = load_data(cfg)
    plot_feature_influence(corr, cfg, out_dir)
    print_summary(corr, cfg)
    print(f"\n  Done – chart saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    run_heatmap()
