import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_anomalies_isolation_forest(df_features, contamination=0.02, random_state=42):
    """
    Input: dataframe of numeric features for anomaly detection
    Output: 1 for anomaly, 0 for normal
    
    Note: Features are normalized using StandardScaler before anomaly detection
    to ensure consistent detection across regions with different scales.
    """
    # Normalize features using StandardScaler
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(
        scaler.fit_transform(df_features),
        columns=df_features.columns,
        index=df_features.index
    )
    
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state
    )

    preds = model.fit_predict(df_normalized)
    # IsolationForest: -1 = anomaly, 1 = normal
    return [1 if p == -1 else 0 for p in preds]