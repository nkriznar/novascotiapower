from sklearn.ensemble import IsolationForest


def detect_anomalies_isolation_forest(df_features, contamination=0.02, random_state=42):
    """
    Input: dataframe of numeric features for anomaly detection
    Output: 1 for anomaly, 0 for normal
    """
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state
    )

    preds = model.fit_predict(df_features)
    # IsolationForest: -1 = anomaly, 1 = normal
    return [1 if p == -1 else 0 for p in preds]