"""Train the waste generation XGBoost model and save it for the API to load.

Mirrors the feature engineering in notebooks/00_basemodel.ipynb. XGBoost is
used because it's the best-performing model there (R2=0.81 single split,
CV mean 0.78) - the MLP in 01_deep_learning.ipynb underperforms on this
dataset's 326 rows and isn't worth serving.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "WhatAWaste_City_Level.csv"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

TARGET = "total_msw_total_msw_generated_tons_year"
RAW_FEATURES = [
    "population_population_number_of_people",
    "income_id",
    "region_id",
    "composition_food_organic_waste_percent",
    "composition_paper_cardboard_percent",
    "composition_plastic_percent",
    "composition_glass_percent",
    "composition_metal_percent",
    "composition_other_percent",
]
COMPOSITION_COLS = [c for c in RAW_FEATURES if c.startswith("composition_")]


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df[TARGET] = pd.to_numeric(df[TARGET].astype(str).str.replace(",", "", regex=False), errors="coerce")

    data = df[RAW_FEATURES + [TARGET]].dropna(subset=[TARGET]).copy()
    data["log_population"] = np.log1p(data["population_population_number_of_people"])
    data["log_target"] = np.log1p(data[TARGET])

    imputer = SimpleImputer(strategy="median")
    data[COMPOSITION_COLS] = imputer.fit_transform(data[COMPOSITION_COLS])

    data = pd.get_dummies(data, columns=["income_id", "region_id"], drop_first=True)

    feature_cols = [c for c in data.columns if c not in (TARGET, "log_target", "population_population_number_of_people")]
    return data[feature_cols], data["log_target"], feature_cols, imputer


def main():
    X, y, feature_cols, imputer = load_and_prepare()

    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X, y)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "xgb_waste_model.pkl")
    joblib.dump(imputer, MODELS_DIR / "composition_imputer.pkl")
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    print(f"Trained on {len(X)} rows, {len(feature_cols)} features.")
    print(f"Saved model + imputer + feature_columns.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
