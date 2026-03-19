import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def _annotate_point(ax, x, y, text, xytext=(6, 6), fontsize=8):
    ax.scatter(x, y, s=28, zorder=5)
    ax.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=xytext,
        fontsize=fontsize
    )


def _find_turning_points(y_values):
    turning_indices = []

    if len(y_values) < 3:
        return turning_indices

    for i in range(1, len(y_values) - 1):
        prev_y = y_values[i - 1]
        curr_y = y_values[i]
        next_y = y_values[i + 1]

        if pd.isna(prev_y) or pd.isna(curr_y) or pd.isna(next_y):
            continue

        is_local_max = curr_y > prev_y and curr_y > next_y
        is_local_min = curr_y < prev_y and curr_y < next_y

        if is_local_max or is_local_min:
            turning_indices.append(i)

    return turning_indices


def _label_turning_points(ax, x_values, y_values):
    turning_indices = _find_turning_points(y_values)

    for i in turning_indices:
        y = y_values[i]
        x = x_values[i]

        prev_y = y_values[i - 1]
        next_y = y_values[i + 1]

        if y > prev_y and y > next_y:
            _annotate_point(ax, x, y, f"{y:.1f}", xytext=(5, 6), fontsize=8)
        elif y < prev_y and y < next_y:
            _annotate_point(ax, x, y, f"{y:.1f}", xytext=(5, -12), fontsize=8)


def _remove_iqr_outliers(series):
    s = pd.to_numeric(series, errors="coerce").copy()

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return s

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    s[(s < lower) | (s > upper)] = np.nan
    return s


def plot_region_results(
    df,
    time_col,
    actual_col,
    forecast_col,
    anomaly_col,
    region_name,
    save_path
):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    df[actual_col] = pd.to_numeric(df[actual_col], errors="coerce")
    df[forecast_col] = pd.to_numeric(df[forecast_col], errors="coerce")
    df[anomaly_col] = pd.to_numeric(df[anomaly_col], errors="coerce").fillna(0).astype(int)

    df = df.dropna(subset=[time_col, actual_col, forecast_col])

    df["hour"] = df[time_col].dt.hour
    df["weekday_num"] = df[time_col].dt.dayofweek
    df["month"] = df[time_col].dt.month

    base, ext = os.path.splitext(save_path)
    overall_path = f"{base}_overall{ext}"
    pattern_path = f"{base}_patterns_4in1{ext}"
    anomaly_path = f"{base}_anomaly{ext}"

    # =========================================================
    # 1) Overall forecast vs actual daily mean 
    # =========================================================
    overall_df = df[[time_col, actual_col, forecast_col]].copy()
    overall_df["actual_no_outlier"] = _remove_iqr_outliers(overall_df[actual_col])
    overall_df["forecast_no_outlier"] = _remove_iqr_outliers(overall_df[forecast_col])

    daily_compare = (
        overall_df
        .set_index(time_col)[["actual_no_outlier", "forecast_no_outlier"]]
        .resample("D")
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(16, 6))
    plt.plot(
        daily_compare[time_col],
        daily_compare["actual_no_outlier"],
        label="Actual (Daily Mean)",
        linewidth=1.8
    )
    plt.plot(
        daily_compare[time_col],
        daily_compare["forecast_no_outlier"],
        label="Forecast (Daily Mean)",
        linewidth=1.8
    )
    plt.title(f"Electricity Demand Forecast - {region_name} (Overall Daily Trend)")
    plt.xlabel("Date")
    plt.ylabel("Consumption (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(overall_path, dpi=150)
    plt.close()

    # =========================================================
    # 2) 4-in-1 plot from original actual data
    # =========================================================

    daily_pattern = (
        df.groupby("hour")[actual_col]
        .mean()
        .reindex(range(24))
        .reset_index()
    )
    daily_x = daily_pattern["hour"].tolist()
    daily_y = daily_pattern[actual_col].tolist()

    weekday_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly_pattern = (
        df.groupby("weekday_num")[actual_col]
        .mean()
        .reindex(range(7))
        .reset_index()
    )
    weekly_x = list(range(7))
    weekly_y = weekly_pattern[actual_col].tolist()

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    yearly_pattern = (
        df.groupby("month")[actual_col]
        .mean()
        .reindex(range(1, 13))
        .reset_index()
    )
    yearly_x = list(range(1, 13))
    yearly_y = yearly_pattern[actual_col].tolist()

    trend_df = (
        df.set_index(time_col)[actual_col]
        .resample("D")
        .mean()
        .rolling(window=30, min_periods=1)
        .mean()
        .reset_index()
    )
    trend_x = trend_df[time_col].tolist()
    trend_y = trend_df[actual_col].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"Electricity Demand Pattern Analysis - {region_name}", fontsize=16)

    # Daily
    ax = axes[0, 0]
    ax.plot(daily_x, daily_y, linewidth=2)
    ax.set_title("Daily Pattern (24 Hours)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Consumption (kWh)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    _label_turning_points(ax, daily_x, daily_y)

    # Weekly
    ax = axes[0, 1]
    ax.plot(weekly_x, weekly_y, linewidth=2)
    ax.set_title("Weekly Pattern (7 Days)")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Consumption (kWh)")
    ax.set_xticks(range(7))
    ax.set_xticklabels(weekday_labels)
    ax.grid(True, alpha=0.3)
    _label_turning_points(ax, weekly_x, weekly_y)

    # Yearly
    ax = axes[1, 0]
    ax.plot(yearly_x, yearly_y, linewidth=2)
    ax.set_title("Yearly Pattern (12 Months)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Consumption (kWh)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.grid(True, alpha=0.3)
    _label_turning_points(ax, yearly_x, yearly_y)

    # Trend
    ax = axes[1, 1]
    ax.plot(trend_x, trend_y, linewidth=2)
    ax.set_title("Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Consumption (kWh)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(pattern_path, dpi=150)
    plt.close()

    # =========================================================
    # 3) Anomaly plot
    #    only anomaly points + normal horizontal line
    # =========================================================
    anomaly_df = df[df[anomaly_col] == 1].copy().sort_values(time_col)
    normal_df = df[df[anomaly_col] == 0].copy()
    normal_consumption = normal_df[actual_col].mean()

    plt.figure(figsize=(16, 6))

    if not np.isnan(normal_consumption):
        plt.axhline(
            y=normal_consumption,
            linestyle="--",
            linewidth=1.5,
            label="Normal Consumption"
        )

    if not anomaly_df.empty:
        plt.plot(
            anomaly_df[time_col],
            anomaly_df[actual_col],
            marker="o",
            linewidth=1.3,
            label="Anomaly Points"
        )

    plt.title(f"Anomaly Plot - {region_name}")
    plt.xlabel("Time")
    plt.ylabel("Consumption (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(anomaly_path, dpi=150)
    plt.close()

    print(f"Saved plot: {overall_path}")
    print(f"Saved plot: {pattern_path}")
    print(f"Saved plot: {anomaly_path}")


def generate_all_region_plots(
    csv_path,
    output_dir,
    time_col="timestamp",
    actual_col="consumption",
    forecast_col="yhat",
    anomaly_col="anomaly",
    region_col="region"
):
    df = pd.read_csv(csv_path)
    df[time_col] = pd.to_datetime(df[time_col])

    os.makedirs(output_dir, exist_ok=True)

    for region_name in sorted(df[region_col].dropna().unique()):
        region_df = df[df[region_col] == region_name].copy()
        region_file = _safe_name(region_name)
        save_path = os.path.join(output_dir, f"{region_file}.png")

        plot_region_results(
            df=region_df,
            time_col=time_col,
            actual_col=actual_col,
            forecast_col=forecast_col,
            anomaly_col=anomaly_col,
            region_name=region_name,
            save_path=save_path
        )


if __name__ == "__main__":
    csv_path = "forecast_predictions_all_regions.csv"
    output_dir = "plots"

    generate_all_region_plots(
        csv_path=csv_path,
        output_dir=output_dir,
        time_col="timestamp",
        actual_col="consumption",
        forecast_col="yhat",
        anomaly_col="anomaly",
        region_col="region"
    )