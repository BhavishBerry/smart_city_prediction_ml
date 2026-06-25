"""Train the parking occupancy Random Forest model and save it for the API to load.

Mirrors notebooks/00_basemodel.ipynb (Navish's primary model: Random Forest,
R2=0.997). This is an autoregressive next-reading model - it predicts the next
Occupancy value from the car park's last 3 known readings (lag_1/2/3) plus
calendar features, not from a single snapshot. The API/dashboard ask the user
for those last 3 readings directly, which is how this kind of forecaster is
actually used in production (an operator/sensor feed provides recent history).
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Parking_Birmingham.csv"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

TARGET = "Occupancy"
FEATURES = ["Capacity", "Hour", "DayOfWeek", "Month", "Weekend", "lag_1", "lag_2", "lag_3", "SystemCodeEncoded"]


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df["LastUpdated"] = pd.to_datetime(df["LastUpdated"])
    df = df.sort_values(["SystemCodeNumber", "LastUpdated"])

    df["Hour"] = df["LastUpdated"].dt.hour
    df["DayOfWeek"] = df["LastUpdated"].dt.dayofweek
    df["Month"] = df["LastUpdated"].dt.month
    df["Weekend"] = (df["DayOfWeek"] >= 5).astype(int)

    for lag in [1, 2, 3]:
        df[f"lag_{lag}"] = df.groupby("SystemCodeNumber")["Occupancy"].shift(lag)
    df = df.dropna().reset_index(drop=True)

    le = LabelEncoder()
    df["SystemCodeEncoded"] = le.fit_transform(df["SystemCodeNumber"])

    lot_capacity = df.groupby("SystemCodeNumber")["Capacity"].first().to_dict()

    return df[FEATURES], df[TARGET], le, lot_capacity


def main():
    X, y, le, lot_capacity = load_and_prepare()

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "rf_parking_model.pkl")
    joblib.dump(le, MODELS_DIR / "system_code_encoder.pkl")
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(FEATURES, f)
    with open(MODELS_DIR / "lot_capacity.json", "w") as f:
        json.dump(lot_capacity, f)

    print(f"Trained on {len(X)} rows, {len(FEATURES)} features, {len(lot_capacity)} parking lots.")
    print(f"Saved model + encoder + feature_columns.json + lot_capacity.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
