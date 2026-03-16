import os
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from .forecasting_model import train_prophet_model, make_forecasts
from .anomaly_detection import detect_anomalies_isolation_forest
from .visualization import plot_region_results


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_forecast(actual, predicted):
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

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
    """
    Save Prophet component plots:
    trend, weekly, yearly, daily
    """
    safe_region = str(region_name).replace("/", "_").replace("\\", "_").replace(" ", "_")

    fig = model.plot_components(forecast)
    fig.savefig(
        os.path.join(plots_dir, f"prophet_components_{safe_region}.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(fig)


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

    print("=" * 60)
    print("Multi-region forecasting pipeline started")
    print("=" * 60)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df[time_col] = pd.to_datetime(train_df[time_col])
    test_df[time_col] = pd.to_datetime(test_df[time_col])

    train_regions = set(train_df[region_col].dropna().unique())
    test_regions = set(test_df[region_col].dropna().unique())
    common_regions = sorted(train_regions.intersection(test_regions))

    print(f"Regions found: {common_regions}")

    all_predictions = []
    metrics_rows = []

    for region in common_regions:
        print(f"\nProcessing region: {region}")

        train_region = train_df[train_df[region_col] == region].copy()
        test_region = test_df[test_df[region_col] == region].copy()

        if train_region.empty or test_region.empty:
            print(f"Skipping {region}: no data")
            continue

        prophet_train = train_region[[time_col, target_col]].rename(
            columns={time_col: "ds", target_col: "y"}
        ).sort_values("ds")

        prophet_test = test_region[[time_col, target_col]].rename(
            columns={time_col: "ds", target_col: "y"}
        ).sort_values("ds")

        model = train_prophet_model(prophet_train)
        forecast = make_forecasts(model, prophet_test[["ds"]])

        # save Prophet component plots
        save_prophet_component_plots(model, forecast, region, plots_dir)

        # keep important Prophet output columns
        component_cols = ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]
        optional_cols = ["daily", "weekly", "yearly", "additive_terms", "multiplicative_terms"]

        for col in optional_cols:
            if col in forecast.columns:
                component_cols.append(col)

        forecast_small = forecast[component_cols].copy()

        result = prophet_test.merge(
            forecast_small,
            on="ds",
            how="left"
        )

        result[region_col] = region
        result["residual"] = result["y"] - result["yhat"]

        result["anomaly"] = detect_anomalies_isolation_forest(
            result[["residual"]].fillna(0)
        )

        mae, rmse, mape = evaluate_forecast(result["y"], result["yhat"])

        metrics_rows.append({
            "region": region,
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        })

        result = result.rename(columns={
            "ds": time_col,
            "y": target_col
        })

        safe_region = str(region).replace("/", "_").replace("\\", "_").replace(" ", "_")

        region_csv_path = os.path.join(output_dir, f"forecast_{safe_region}.csv")
        result.to_csv(region_csv_path, index=False)

        plot_file = os.path.join(plots_dir, f"forecast_{safe_region}.png")
        plot_region_results(
            df=result,
            time_col=time_col,
            actual_col=target_col,
            forecast_col="yhat",
            anomaly_col="anomaly",
            region_name=region,
            save_path=plot_file
        )

        all_predictions.append(result)

        print(f"Finished region: {region}")
        print(f"MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
        print(f"Forecast CSV saved: {region_csv_path}")

        available_components = [c for c in ["trend", "daily", "weekly", "yearly"] if c in result.columns]
        print(f"Saved Prophet components for {region}: {available_components}")

    if all_predictions:
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        combined_forecast_path = os.path.join(output_dir, "forecast_predictions_all_regions.csv")
        predictions_df.to_csv(combined_forecast_path, index=False)
        print(f"\nCombined forecast CSV saved: {combined_forecast_path}")
    else:
        predictions_df = pd.DataFrame()
        print("\nNo prediction results generated.")

    metrics_df = pd.DataFrame(metrics_rows)

    if not metrics_df.empty:
        overall_row = {
            "region": "OVERALL_AVG",
            "mae": metrics_df["mae"].mean(),
            "rmse": metrics_df["rmse"].mean(),
            "mape": metrics_df["mape"].mean()
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([overall_row])], ignore_index=True)

        metrics_path = os.path.join(output_dir, "forecast_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)

        print(f"Metrics CSV saved: {metrics_path}")
        print("\nMetrics summary:")
        print(metrics_df)
    else:
        print("No metrics generated.")

    print("\nPipeline finished.")


if __name__ == "__main__":
    train_and_evaluate_model()