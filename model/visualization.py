import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_region_results(df, time_col, actual_col, forecast_col, anomaly_col, region_name, save_path):
    """
    Create 2 plots for each region:

    1. Overall trend plot
       - daily mean actual
       - daily mean forecast
       - anomaly days highlighted

    2. Last 30 days detailed plot
       - hourly actual
       - hourly forecast
       - anomaly points highlighted
       - anomaly points labeled with date + time

    Files saved:
      <name>_overall.png
      <name>_last30days.png
    """

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    # make sure anomaly column is numeric
    df[anomaly_col] = pd.to_numeric(df[anomaly_col], errors="coerce").fillna(0).astype(int)

    # -----------------------------
    # 1) OVERALL PLOT (daily mean)
    # -----------------------------
    daily_lines = (
        df.set_index(time_col)[[actual_col, forecast_col]]
        .resample("D")
        .mean()
        .reset_index()
    )

    anomaly_df = df[df[anomaly_col] == 1].copy()

    if not anomaly_df.empty:
        daily_anomaly = (
            anomaly_df.set_index(time_col)
            .resample("D")
            .agg(
                anomaly_count=(anomaly_col, "sum"),
                anomaly_peak=(actual_col, "max")
            )
            .reset_index()
        )
        daily_anomaly = daily_anomaly[daily_anomaly["anomaly_count"] > 0]
    else:
        daily_anomaly = pd.DataFrame(columns=[time_col, "anomaly_count", "anomaly_peak"])

    base, ext = os.path.splitext(save_path)
    overall_path = f"{base}_overall{ext}"

    plt.figure(figsize=(16, 6))
    plt.plot(
        daily_lines[time_col],
        daily_lines[actual_col],
        label="Actual (Daily Mean)",
        linewidth=1.8
    )
    plt.plot(
        daily_lines[time_col],
        daily_lines[forecast_col],
        label="Forecast (Daily Mean)",
        linewidth=1.8
    )

    if not daily_anomaly.empty:
        plt.scatter(
            daily_anomaly[time_col],
            daily_anomaly["anomaly_peak"],
            label="Days with Anomalies",
            s=28
        )

    plt.title(f"Electricity Demand Forecast and Anomalies - {region_name} (Overall Daily Trend)")
    plt.xlabel("Time")
    plt.ylabel(actual_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(overall_path, dpi=150)
    plt.close()

    # --------------------------------
    # 2) LAST 30 DAYS DETAILED PLOT
    # --------------------------------
    max_time = df[time_col].max()
    recent_start = max_time - pd.Timedelta(days=30)

    recent_df = df[df[time_col] >= recent_start].copy()
    recent_anomalies = recent_df[recent_df[anomaly_col] == 1].copy()

    recent_path = f"{base}_last30days{ext}"

    plt.figure(figsize=(16, 6))
    plt.plot(
        recent_df[time_col],
        recent_df[actual_col],
        label="Actual",
        linewidth=1.0
    )
    plt.plot(
        recent_df[time_col],
        recent_df[forecast_col],
        label="Forecast",
        linewidth=1.2
    )

    if not recent_anomalies.empty:
        plt.scatter(
            recent_anomalies[time_col],
            recent_anomalies[actual_col],
            label="Anomalies",
            s=22
        )

        # label anomaly points with date and time
        for _, row in recent_anomalies.iterrows():
            label_text = row[time_col].strftime("%Y-%m-%d %H:%M")
            plt.annotate(
                label_text,
                (row[time_col], row[actual_col]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7
            )

    plt.title(f"Electricity Demand Forecast and Anomalies - {region_name} (Last 30 Days)")
    plt.xlabel("Time")
    plt.ylabel(actual_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(recent_path, dpi=150)
    plt.close()

    print(f"Saved plot: {overall_path}")
    print(f"Saved plot: {recent_path}")