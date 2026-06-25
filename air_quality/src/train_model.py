"""Train the CO(GT) Random Forest model and save it for the API to load.

Mirrors notebooks/00_basemodel.ipynb (Rishu's Random Forest regressor, R2=0.939
on the original notebook's scaled-feature run; here unscaled since RF doesn't
need it, only her SVR/Linear Regression cells did).

Also exposes classify_air_quality(), the same Good/Moderate/Poor bucketing she
used for the classification half of her notebook, so the API can return a
human-readable category alongside the raw CO(GT) prediction.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "AirQualityUCI.csv"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

TARGET = "CO(GT)"


def classify_air_quality(co: float) -> str:
    if co <= 2.0:
        return "Good"
    elif co <= 5.0:
        return "Moderate"
    return "Poor"


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["NMHC(GT)"])
    df = df.dropna()

    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
    df.drop(["Date", "Time"], axis=1, inplace=True)

    feature_cols = [c for c in df.columns if c != TARGET]
    return df[feature_cols], df[TARGET], feature_cols


def main():
    X, y, feature_cols = load_and_prepare()

    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X, y)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "rf_air_quality_model.pkl")
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    print(f"Trained on {len(X)} rows, {len(feature_cols)} features.")
    print(f"Saved model + feature_columns.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
