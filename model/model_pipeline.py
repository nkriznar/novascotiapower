import os
import time
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from model.forecasting_model import (
    train_prophet_model,
    make_forecasts,
    train_arima_and_forecast
)
from model.anomaly_detection import detect_anomalies_isolation_forest
from model.visualization import plot_region_results


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_forecast(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    nonzero_mask = actual != 0
    if np.any(nonzero_mask):
        mape = np.mean(
            np.abs((actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask])
        ) * 100
    else:
        mape = np.nan

    return mae, rmse, mape


def save_prophet_component_plots(model, forecast, region_name, plots_dir):
    safe_region = str(region_name).replace("/", "_").replace("\\", "_").replace(" ", "_")

    fig = model.plot_components(forecast)
    axes = fig.get_axes()

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=10)

    if len(axes) >= 1:
        axes[0].set_title(f"Trend Component - {region_name}", fontsize=14)
        axes[0].set_xlabel("Date", fontsize=11)
        axes[0].set_ylabel("Electricity Demand (kWh)", fontsize=11)

    if len(axes) >= 2:
        axes[1].set_title("Weekly Pattern", fontsize=14)
        axes[1].set_xlabel("Day of Week", fontsize=11)
        axes[1].set_ylabel("Effect on Demand (kWh)", fontsize=11)

    if len(axes) >= 3:
        axes[2].set_title("Yearly Pattern", fontsize=14)
        axes[2].set_xlabel("Month of Year", fontsize=11)
        axes[2].set_ylabel("Effect on Demand (kWh)", fontsize=11)

    if len(axes) >= 4:
        axes[3].set_title("Daily Pattern", fontsize=14)
        axes[3].set_xlabel("Hour of Day", fontsize=11)
        axes[3].set_ylabel("Effect on Demand (kWh)", fontsize=11)

    fig.set_size_inches(12, 14)
    plt.tight_layout()
    fig.savefig(
        os.path.join(plots_dir, f"prophet_components_{safe_region}.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


def run_pipeline(train_df, test_df, target_col, time_col, region_col, output_dir, plots_dir):
    train_regions = set(train_df[region_col].dropna().unique())
    test_regions = set(test_df[region_col].dropna().unique())
    common_regions = sorted(train_regions.intersection(test_regions))

    print("=" * 60)
    print("Pipeline started: prophet outputs + arima metrics only")
    print("=" * 60)
    print(f"Regions found: {common_regions}")

    prophet_predictions = []
    metrics_rows = []

    for region in common_regions:
        region_start = time.perf_counter()
        print(f"\nProcessing region: {region}")

        train_region = train_df[train_df[region_col] == region].copy().sort_values(time_col)
        test_region = test_df[test_df[region_col] == region].copy().sort_values(time_col)

        if train_region.empty or test_region.empty:
            print(f"Skipping {region}: no data")
            continue

        # =====================================================
        # 1) PROPHET 
        # =====================================================
        prophet_train = (
            train_region[[time_col, target_col]]
            .rename(columns={time_col: "ds", target_col: "y"})
            .sort_values("ds")
        )

        prophet_test = (
            test_region[[time_col, target_col]]
            .rename(columns={time_col: "ds", target_col: "y"})
            .sort_values("ds")
        )

        print(
            f"Starting Prophet training for {region} "
            f"(rows={len(prophet_train):,}, cap=20,000)..."
        )
        prophet_fit_start = time.perf_counter()
        prophet_model = train_prophet_model(prophet_train, max_points=20000)
        prophet_fit_elapsed = time.perf_counter() - prophet_fit_start
        print(f"Prophet training finished for {region} in {prophet_fit_elapsed:.1f}s")

        print(f"Starting Prophet forecast for {region} (horizon={len(prophet_test):,})...")
        prophet_forecast = make_forecasts(prophet_model, prophet_test[["ds"]])
        print(f"Prophet forecast finished for {region}")

        save_prophet_component_plots(prophet_model, prophet_forecast, region, plots_dir)

        keep_cols = ["ds", "yhat"]
        for col in [
            "yhat_lower",
            "yhat_upper",
            "trend",
            "daily",
            "weekly",
            "yearly",
            "additive_terms",
            "multiplicative_terms"
        ]:
            if col in prophet_forecast.columns:
                keep_cols.append(col)

        result = prophet_test.merge(prophet_forecast[keep_cols], on="ds", how="left")
        result[region_col] = region
        result["model_type"] = "prophet"
        result["residual"] = result["y"] - result["yhat"]
        result["abs_residual"] = result["residual"].abs()
        result["anomaly"] = detect_anomalies_isolation_forest(result[["residual"]].fillna(0))

        prophet_mae, prophet_rmse, prophet_mape = evaluate_forecast(result["y"], result["yhat"])
        metrics_rows.append({
            "region": region,
            "model_type": "prophet",
            "mae": prophet_mae,
            "rmse": prophet_rmse,
            "mape": prophet_mape
        })

        result = result.rename(columns={"ds": time_col, "y": target_col})

        safe_region = str(region).replace("/", "_").replace("\\", "_").replace(" ", "_")

        # Prophet region csv
        region_csv_path = os.path.join(output_dir, f"forecast_{safe_region}.csv")
        result.to_csv(region_csv_path, index=False)

        # Prophet plots
        plot_region_results(
            df=result,
            time_col=time_col,
            actual_col=target_col,
            forecast_col="yhat",
            anomaly_col="anomaly",
            region_name=region,
            save_path=os.path.join(plots_dir, f"forecast_{safe_region}.png")
        )

        prophet_predictions.append(result)

        print(f"Prophet done: {region}")
        print(f"Prophet MAE={prophet_mae:.4f}, RMSE={prophet_rmse:.4f}, MAPE={prophet_mape:.2f}%")

        # =====================================================
        # 2) ARIMA 
        # =====================================================
        try:
            arima_train = pd.to_numeric(train_region[target_col], errors="coerce").dropna().values
            arima_test = pd.to_numeric(test_region[target_col], errors="coerce").dropna().values

            if len(arima_train) >= 20 and len(arima_test) > 0:
                print(
                    f"Starting ARIMA for {region} "
                    f"(train={len(arima_train):,}, capped=5,000, horizon={len(arima_test):,})..."
                )
                arima_start = time.perf_counter()
                arima_pred, best_order = train_arima_and_forecast(
                    train_series=arima_train,
                    steps=len(arima_test),
                    max_p=2,
                    max_d=1,
                    max_q=2,
                    max_train_points=5000,
                    maxiter=40
                )
                arima_elapsed = time.perf_counter() - arima_start

                arima_mae, arima_rmse, arima_mape = evaluate_forecast(arima_test, arima_pred)

                metrics_rows.append({
                    "region": region,
                    "model_type": "arima",
                    "mae": arima_mae,
                    "rmse": arima_rmse,
                    "mape": arima_mape
                })

                print(f"ARIMA done: {region}")
                print(f"Best ARIMA order={best_order}")
                print(f"ARIMA MAE={arima_mae:.4f}, RMSE={arima_rmse:.4f}, MAPE={arima_mape:.2f}%")
                print(f"ARIMA runtime for {region}: {arima_elapsed:.1f}s")
            else:
                print(f"Skipping ARIMA for {region}: insufficient data")

        except Exception as e:
            print(f"ARIMA failed for {region}: {e}")

        region_elapsed = time.perf_counter() - region_start
        print(f"Region completed: {region} in {region_elapsed:.1f}s")

    # =========================================================
    # Save PROPHET outputs only
    # =========================================================
    if prophet_predictions:
        predictions_df = pd.concat(prophet_predictions, ignore_index=True)

        predictions_df.to_csv(
            os.path.join(output_dir, "forecast_predictions_all_regions.csv"),
            index=False
        )

        anomaly_events = predictions_df[predictions_df["anomaly"] == 1].copy()
        anomaly_events.to_csv(
            os.path.join(output_dir, "anomaly_events.csv"),
            index=False
        )

        anomaly_summary = (
            predictions_df.groupby(region_col)
            .agg(
                total_points=("anomaly", "count"),
                anomaly_count=("anomaly", "sum")
            )
            .reset_index()
        )
        anomaly_summary["anomaly_rate"] = (
            anomaly_summary["anomaly_count"] / anomaly_summary["total_points"]
        )

        anomaly_summary.to_csv(
            os.path.join(output_dir, "anomaly_summary_by_region.csv"),
            index=False
        )

    # =========================================================
    # One summary metrics file for comparison only
    # =========================================================
    metrics_df = pd.DataFrame(metrics_rows)

    if not metrics_df.empty:
        overall_rows = []
        for model_name in ["prophet", "arima"]:
            temp = metrics_df[metrics_df["model_type"] == model_name]
            if not temp.empty:
                overall_rows.append({
                    "region": "OVERALL_AVG",
                    "model_type": model_name,
                    "mae": temp["mae"].mean(),
                    "rmse": temp["rmse"].mean(),
                    "mape": temp["mape"].mean()
                })

        metrics_summary_df = pd.concat(
            [metrics_df, pd.DataFrame(overall_rows)],
            ignore_index=True
        )

        metrics_summary_df = metrics_summary_df.sort_values(
            by=["region", "model_type"]
        ).reset_index(drop=True)

        metrics_summary_df.to_csv(
            os.path.join(output_dir, "forecast_metrics_summary.csv"),
            index=False
        )

        print("\nMetrics summary:")
        print(metrics_summary_df)

    print("\nPipeline finished")
    print("Prophet: outputs + plots + csv")
    print("ARIMA: metrics only")


def train_and_evaluate_model():
    config = load_config()

    train_path = config["paths"]["train_output"]
    test_path = config["paths"]["test_output"]
    output_dir = config["paths"].get("output_dir", "outputs")
    plots_dir = os.path.join(output_dir, "plots")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    target_col = config["columns"]["target"]
    time_col = config["columns"]["timestamp"]
    region_col = config["columns"]["region"]

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df[time_col] = pd.to_datetime(train_df[time_col])
    test_df[time_col] = pd.to_datetime(test_df[time_col])

    run_pipeline(
        train_df=train_df,
        test_df=test_df,
        target_col=target_col,
        time_col=time_col,
        region_col=region_col,
        output_dir=output_dir,
        plots_dir=plots_dir
    )


if __name__ == "__main__":
    train_and_evaluate_model()