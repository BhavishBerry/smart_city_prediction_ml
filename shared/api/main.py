"""Shared prediction API for all 4 smart-city domains.

Each domain has a train_model.py under <domain>/src/ that mirrors that
person's best model from their notebooks/00_basemodel.ipynb. Model artifacts
are gitignored (binary, regenerable), so every loader below auto-trains on
first run if its .pkl is missing - a fresh clone works with no manual setup.
"""

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Smart City Prediction API")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _ensure_trained(domain: str, marker_file: str):
    models_dir = REPO_ROOT / domain / "models"
    if not (models_dir / marker_file).exists():
        # Each domain has its own train_model.py, all with the same module
        # name - load each under a unique name so they don't shadow each
        # other in sys.modules.
        script_path = REPO_ROOT / domain / "src" / "train_model.py"
        spec = importlib.util.spec_from_file_location(f"{domain}_train_model", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
    return models_dir


# ---------------------------------------------------------------------------
# Waste (Bhavish) - XGBoost, R2=0.81 single split / 0.78 +/-0.07 CV
# ---------------------------------------------------------------------------
_WASTE_DIR = _ensure_trained("waste", "xgb_waste_model.pkl")
_waste_model = joblib.load(_WASTE_DIR / "xgb_waste_model.pkl")
_waste_imputer = joblib.load(_WASTE_DIR / "composition_imputer.pkl")
with open(_WASTE_DIR / "feature_columns.json") as f:
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


@app.post("/predict/waste", response_model=WastePrediction)
def predict_waste(payload: WasteInput):
    X = _build_waste_feature_row(payload)
    pred_log = _waste_model.predict(X)[0]
    return WastePrediction(predicted_waste_tons_per_year=float(np.expm1(pred_log)))


# ---------------------------------------------------------------------------
# Traffic (Anshika) - Random Forest, R2=0.96
# ---------------------------------------------------------------------------
_TRAFFIC_DIR = _ensure_trained("traffic", "rf_traffic_model.pkl")
_traffic_model = joblib.load(_TRAFFIC_DIR / "rf_traffic_model.pkl")
with open(_TRAFFIC_DIR / "feature_columns.json") as f:
    _traffic_feature_columns = json.load(f)

_HOLIDAYS = sorted({c.split("holiday_", 1)[1] for c in _traffic_feature_columns if c.startswith("holiday_")})
_WEATHER_MAIN = sorted({c.split("weather_main_", 1)[1] for c in _traffic_feature_columns if c.startswith("weather_main_")})
_WEATHER_DESC = sorted({c.split("weather_description_", 1)[1] for c in _traffic_feature_columns if c.startswith("weather_description_")})


class TrafficInput(BaseModel):
    temp: float = Field(..., description="Temperature in Kelvin")
    rain_1h: float = 0.0
    snow_1h: float = 0.0
    clouds_all: float = Field(..., ge=0, le=100, description="Cloud cover %")
    hour: int = Field(..., ge=0, le=23)
    day: int = Field(..., ge=1, le=31)
    month: int = Field(..., ge=1, le=12)
    dayofweek: int = Field(..., ge=0, le=6, description="0=Monday")
    holiday: str | None = Field(None, description=f"None or one of {_HOLIDAYS}")
    weather_main: str = Field(..., description=f"one of {_WEATHER_MAIN}")
    weather_description: str = Field(..., description=f"one of {_WEATHER_DESC}")


class TrafficPrediction(BaseModel):
    predicted_traffic_volume: float


@app.post("/predict/traffic", response_model=TrafficPrediction)
def predict_traffic(payload: TrafficInput):
    row = {col: 0.0 for col in _traffic_feature_columns}
    row["temp"] = payload.temp
    row["rain_1h"] = payload.rain_1h
    row["snow_1h"] = payload.snow_1h
    row["clouds_all"] = payload.clouds_all
    row["hour"] = payload.hour
    row["day"] = payload.day
    row["month"] = payload.month
    row["dayofweek"] = payload.dayofweek

    if payload.holiday:
        col = f"holiday_{payload.holiday}"
        if col not in row:
            raise HTTPException(400, f"holiday must be None or one of {_HOLIDAYS}")
        row[col] = 1.0

    weather_main_col = f"weather_main_{payload.weather_main}"
    if weather_main_col not in row:
        raise HTTPException(400, f"weather_main must be one of {_WEATHER_MAIN}")
    row[weather_main_col] = 1.0

    weather_desc_col = f"weather_description_{payload.weather_description}"
    if weather_desc_col not in row:
        raise HTTPException(400, f"weather_description must be one of {_WEATHER_DESC}")
    row[weather_desc_col] = 1.0

    X = pd.DataFrame([row], columns=_traffic_feature_columns)
    pred = _traffic_model.predict(X)[0]
    return TrafficPrediction(predicted_traffic_volume=float(pred))


# ---------------------------------------------------------------------------
# Air quality (Rishu) - Random Forest, R2=0.939
# ---------------------------------------------------------------------------
_AIR_DIR = _ensure_trained("air_quality", "rf_air_quality_model.pkl")
_air_model = joblib.load(_AIR_DIR / "rf_air_quality_model.pkl")
with open(_AIR_DIR / "feature_columns.json") as f:
    _air_feature_columns = json.load(f)


def _classify_air_quality(co: float) -> str:
    if co <= 2.0:
        return "Good"
    elif co <= 5.0:
        return "Moderate"
    return "Poor"


class AirQualityInput(BaseModel):
    PT08_S1_CO: float = Field(..., alias="PT08.S1(CO)")
    C6H6_GT: float = Field(..., alias="C6H6(GT)")
    PT08_S2_NMHC: float = Field(..., alias="PT08.S2(NMHC)")
    NOx_GT: float = Field(..., alias="NOx(GT)")
    PT08_S3_NOx: float = Field(..., alias="PT08.S3(NOx)")
    NO2_GT: float = Field(..., alias="NO2(GT)")
    PT08_S4_NO2: float = Field(..., alias="PT08.S4(NO2)")
    PT08_S5_O3: float = Field(..., alias="PT08.S5(O3)")
    T: float
    RH: float
    AH: float
    Day: int = Field(..., ge=1, le=31)
    Month: int = Field(..., ge=1, le=12)
    Year: int
    Hour: int = Field(..., ge=0, le=23)

    class Config:
        populate_by_name = True


class AirQualityPrediction(BaseModel):
    predicted_co_gt: float
    air_quality_category: str


@app.post("/predict/air_quality", response_model=AirQualityPrediction)
def predict_air_quality(payload: AirQualityInput):
    values = {
        "PT08.S1(CO)": payload.PT08_S1_CO,
        "C6H6(GT)": payload.C6H6_GT,
        "PT08.S2(NMHC)": payload.PT08_S2_NMHC,
        "NOx(GT)": payload.NOx_GT,
        "PT08.S3(NOx)": payload.PT08_S3_NOx,
        "NO2(GT)": payload.NO2_GT,
        "PT08.S4(NO2)": payload.PT08_S4_NO2,
        "PT08.S5(O3)": payload.PT08_S5_O3,
        "T": payload.T,
        "RH": payload.RH,
        "AH": payload.AH,
        "Day": payload.Day,
        "Month": payload.Month,
        "Year": payload.Year,
        "Hour": payload.Hour,
    }
    X = pd.DataFrame([values], columns=_air_feature_columns)
    pred = float(_air_model.predict(X)[0])
    return AirQualityPrediction(predicted_co_gt=pred, air_quality_category=_classify_air_quality(pred))


# ---------------------------------------------------------------------------
# Parking (Navish) - Random Forest, R2=0.997 (autoregressive: needs last 3 readings)
# ---------------------------------------------------------------------------
_PARKING_DIR = _ensure_trained("parking", "rf_parking_model.pkl")
_parking_model = joblib.load(_PARKING_DIR / "rf_parking_model.pkl")
_parking_encoder = joblib.load(_PARKING_DIR / "system_code_encoder.pkl")
with open(_PARKING_DIR / "feature_columns.json") as f:
    _parking_feature_columns = json.load(f)
with open(_PARKING_DIR / "lot_capacity.json") as f:
    _lot_capacity = json.load(f)

_PARKING_LOTS = sorted(_lot_capacity.keys())


class ParkingInput(BaseModel):
    system_code: str = Field(..., description=f"one of {_PARKING_LOTS}")
    hour: int = Field(..., ge=0, le=23)
    dayofweek: int = Field(..., ge=0, le=6, description="0=Monday")
    month: int = Field(..., ge=1, le=12)
    lag_1: float = Field(..., description="most recent known occupancy reading")
    lag_2: float = Field(..., description="2nd most recent known occupancy reading")
    lag_3: float = Field(..., description="3rd most recent known occupancy reading")
    capacity: float | None = Field(None, description="defaults to the lot's known capacity if omitted")


class ParkingPrediction(BaseModel):
    predicted_occupancy: float
    capacity_used: float
    predicted_occupancy_rate: float


@app.post("/predict/parking", response_model=ParkingPrediction)
def predict_parking(payload: ParkingInput):
    if payload.system_code not in _lot_capacity:
        raise HTTPException(400, f"system_code must be one of {_PARKING_LOTS}")

    capacity = payload.capacity if payload.capacity is not None else _lot_capacity[payload.system_code]
    encoded = int(_parking_encoder.transform([payload.system_code])[0])
    weekend = int(payload.dayofweek >= 5)

    row = {
        "Capacity": capacity,
        "Hour": payload.hour,
        "DayOfWeek": payload.dayofweek,
        "Month": payload.month,
        "Weekend": weekend,
        "lag_1": payload.lag_1,
        "lag_2": payload.lag_2,
        "lag_3": payload.lag_3,
        "SystemCodeEncoded": encoded,
    }
    X = pd.DataFrame([row], columns=_parking_feature_columns)
    pred = float(_parking_model.predict(X)[0])
    return ParkingPrediction(
        predicted_occupancy=pred,
        capacity_used=capacity,
        predicted_occupancy_rate=pred / capacity if capacity else 0.0,
    )


@app.get("/")
def root():
    return {"status": "ok", "domains": ["waste", "traffic", "air_quality", "parking"]}
