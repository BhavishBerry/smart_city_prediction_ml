"""Shared prediction API for all 4 smart-city domains.

Only /predict/waste is wired up for real right now - it's the only domain
with a saved, trained model artifact (waste/models/, built by
waste/src/train_model.py). The other three return 501 until their owners
add an equivalent train_model.py + saved model in their own <domain>/models/
folder and someone wires up the matching endpoint here.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Smart City Prediction API")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WASTE_MODELS_DIR = REPO_ROOT / "waste" / "models"

if not (WASTE_MODELS_DIR / "xgb_waste_model.pkl").exists():
    # Model artifacts are gitignored (binary, regenerable) - train on first run
    # instead of requiring a manual step before the API can start.
    import sys

    sys.path.insert(0, str(REPO_ROOT / "waste" / "src"))
    from train_model import main as train_waste_model

    train_waste_model()

_waste_model = joblib.load(WASTE_MODELS_DIR / "xgb_waste_model.pkl")
_waste_imputer = joblib.load(WASTE_MODELS_DIR / "composition_imputer.pkl")
with open(WASTE_MODELS_DIR / "feature_columns.json") as f:
    _waste_feature_columns = json.load(f)

_INCOME_IDS = ["HIC", "LIC", "LMC", "UMC"]
_REGION_IDS = ["EAS", "ECS", "LCN", "MEA", "NAC", "SAS", "SSF"]
_COMPOSITION_COLS = [
    "composition_food_organic_waste_percent",
    "composition_paper_cardboard_percent",
    "composition_plastic_percent",
    "composition_glass_percent",
    "composition_metal_percent",
    "composition_other_percent",
]


class WasteInput(BaseModel):
    population: float = Field(..., gt=0, description="City population")
    income_id: str = Field(..., description=f"World Bank income group, one of {_INCOME_IDS}")
    region_id: str = Field(..., description=f"World Bank region, one of {_REGION_IDS}")
    composition_food_organic_waste_percent: float | None = None
    composition_paper_cardboard_percent: float | None = None
    composition_plastic_percent: float | None = None
    composition_glass_percent: float | None = None
    composition_metal_percent: float | None = None
    composition_other_percent: float | None = None


class WastePrediction(BaseModel):
    predicted_waste_tons_per_year: float


def _build_waste_feature_row(payload: WasteInput) -> pd.DataFrame:
    if payload.income_id not in _INCOME_IDS:
        raise HTTPException(400, f"income_id must be one of {_INCOME_IDS}")
    if payload.region_id not in _REGION_IDS:
        raise HTTPException(400, f"region_id must be one of {_REGION_IDS}")

    composition = pd.DataFrame([[getattr(payload, c) for c in _COMPOSITION_COLS]], columns=_COMPOSITION_COLS)
    composition[:] = _waste_imputer.transform(composition)

    row = {col: 0.0 for col in _waste_feature_columns}
    row["log_population"] = np.log1p(payload.population)
    for col in _COMPOSITION_COLS:
        row[col] = composition[col].iloc[0]

    income_col = f"income_id_{payload.income_id}"
    if income_col in row:
        row[income_col] = 1.0
    region_col = f"region_id_{payload.region_id}"
    if region_col in row:
        row[region_col] = 1.0

    return pd.DataFrame([row], columns=_waste_feature_columns)


@app.get("/")
def root():
    return {"status": "ok", "domains": ["waste", "traffic", "air_quality", "parking"]}


@app.post("/predict/waste", response_model=WastePrediction)
def predict_waste(payload: WasteInput):
    X = _build_waste_feature_row(payload)
    pred_log = _waste_model.predict(X)[0]
    return WastePrediction(predicted_waste_tons_per_year=float(np.expm1(pred_log)))


@app.post("/predict/traffic")
def predict_traffic():
    raise HTTPException(501, "Traffic model not wired up yet - needs traffic/models/ + a loader here.")


@app.post("/predict/air_quality")
def predict_air_quality():
    raise HTTPException(501, "Air quality model not wired up yet - needs air_quality/models/ + a loader here.")


@app.post("/predict/parking")
def predict_parking():
    raise HTTPException(501, "Parking model not wired up yet - needs parking/models/ + a loader here.")
