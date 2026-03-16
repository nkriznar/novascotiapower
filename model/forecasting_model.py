from prophet import Prophet


def train_prophet_model(train_df):
    """
    train_df must contain:
    ds = datetime column
    y  = target column
    """
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95
    )

    model.fit(train_df)
    return model


def make_forecasts(model, future_df):
    """
    future_df must contain a ds column.
    Returns full Prophet forecast including component columns:
    yhat, trend, daily, weekly, yearly, etc.
    """
    forecast = model.predict(future_df)
    return forecast