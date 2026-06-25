"""Train the traffic volume Random Forest model and save it for the API to load.

Mirrors notebooks/00_basemodel.ipynb (Anshika's best model: Random Forest, R2=0.96).
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Metro_Interstate_Traffic_Volume.csv"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

TARGET = "traffic_volume"
DUMMY_COLUMNS = ["holiday", "weather_main", "weather_description"]


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df.drop_duplicates(inplace=True)

    df = pd.get_dummies(df, columns=DUMMY_COLUMNS, drop_first=True)

    df["date_time"] = pd.to_datetime(df["date_time"])
    df["hour"] = df["date_time"].dt.hour
    df["day"] = df["date_time"].dt.day
    df["month"] = df["date_time"].dt.month
    df["dayofweek"] = df["date_time"].dt.dayofweek
    df.drop("date_time", axis=1, inplace=True)

    feature_cols = [c for c in df.columns if c != TARGET]
    return df[feature_cols], df[TARGET], feature_cols


def main():
    X, y, feature_cols = load_and_prepare()

    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X, y)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "rf_traffic_model.pkl")
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    print(f"Trained on {len(X)} rows, {len(feature_cols)} features.")
    print(f"Saved model + feature_columns.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
