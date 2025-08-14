import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def make_features(df: pd.DataFrame):
    df = df.copy()
    # Target: next-day up (1) or down (0)
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    feature_cols = [
        "Return","SMA_10","SMA_20","Momentum_5","Volatility_10","Volume_Change"
    ]

    X = df[feature_cols].copy()
    # Replace inf/-inf and fill any residual NaNs
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df["Target"].copy()

    # Drop last row (no target due to shift)
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    return X, y, feature_cols

def train_baseline(X_train, y_train, X_test, y_test):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200))
    ])
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:,1]
    y_pred = (y_prob > 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "report": classification_report(y_test, y_pred, output_dict=False)
    }
    return pipe, metrics, y_prob
