from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
import math
import warnings


def _downsample_for_prophet(train_df, max_points):
    """
    Keep the full time span while reducing point count for very large histories.
    """
    if len(train_df) <= max_points:
        return train_df

    step = math.ceil(len(train_df) / max_points)
    sampled = train_df.iloc[::step].copy()

    # Guarantee the latest observation is always present.
    if sampled.iloc[-1]["ds"] != train_df.iloc[-1]["ds"]:
        sampled = pd.concat([sampled, train_df.tail(1)], axis=0)

    return sampled.drop_duplicates(subset=["ds"]).sort_values("ds")


def train_prophet_model(train_df, max_points=20000):
    """
    Baseline Prophet model.
    train_df must contain:
        ds = datetime column
        y  = target column
    """
    train_df = train_df[["ds", "y"]].dropna().sort_values("ds")
    train_df = _downsample_for_prophet(train_df, max_points=max_points)

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95,
        uncertainty_samples=0
    )
    model.fit(train_df)
    return model


def make_forecasts(model, future_df):
    """
    Prophet forecast.
    future_df must contain:
        ds
    """
    return model.predict(future_df)


def _safe_mape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = actual != 0
    if np.any(mask):
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return np.inf


def _fit_single_arima(train_series, order, maxiter=40):
    """
    Fit one ARIMA model safely.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = ARIMA(
            train_series,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted = model.fit(method_kwargs={"maxiter": maxiter, "disp": 0})
    return fitted


def select_best_arima_order(train_series, max_p=2, max_d=1, max_q=2, maxiter=40):
    """
    Simple ARIMA grid search using AIC.
    Smaller search to keep it practical.
    """
    best_order = None
    best_aic = np.inf

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                # skip completely empty model
                if p == 0 and d == 0 and q == 0:
                    continue
                try:
                    fitted = _fit_single_arima(train_series, (p, d, q), maxiter=maxiter)
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

    if best_order is None:
        best_order = (1, 1, 1)

    return best_order


def train_arima_and_forecast(
    train_series,
    steps,
    max_p=2,
    max_d=1,
    max_q=2,
    max_train_points=5000,
    maxiter=40
):
    """
    Train ARIMA on one region and forecast for test length.

    Returns:
        forecast_values
        best_order
    """
    train_series = np.asarray(train_series, dtype=float)

    if len(train_series) > max_train_points:
        train_series = train_series[-max_train_points:]

    best_order = select_best_arima_order(
        train_series=train_series,
        max_p=max_p,
        max_d=max_d,
        max_q=max_q,
        maxiter=maxiter
    )

    fitted = _fit_single_arima(train_series, best_order, maxiter=maxiter)
    forecast_values = fitted.forecast(steps=steps)

    return np.asarray(forecast_values, dtype=float), best_order