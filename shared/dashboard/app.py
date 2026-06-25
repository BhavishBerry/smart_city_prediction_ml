"""Smart city prediction dashboard - one tab per domain.

Loads each domain's model directly (no API dependency, so this works
standalone). Model artifacts are gitignored, so every loader auto-trains on
first run if its .pkl is missing.
"""

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def ensure_trained(domain: str, marker_file: str) -> Path:
    models_dir = REPO_ROOT / domain / "models"
    if not (models_dir / marker_file).exists():
        script_path = REPO_ROOT / domain / "src" / "train_model.py"
        spec = importlib.util.spec_from_file_location(f"{domain}_train_model", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
    return models_dir


st.set_page_config(page_title="Smart City Predictions", layout="centered")
st.title("Smart City Predictions")

tab_waste, tab_traffic, tab_air, tab_parking = st.tabs(
    ["Waste", "Traffic", "Air Quality", "Parking"]
)

# ---------------------------------------------------------------------------
# Waste
# ---------------------------------------------------------------------------
INCOME_IDS = ["HIC", "UMC", "LMC", "LIC"]
REGION_IDS = ["EAS", "ECS", "LCN", "MEA", "NAC", "SAS", "SSF"]
COMPOSITION_COLS = [
    "composition_food_organic_waste_percent",
    "composition_paper_cardboard_percent",
    "composition_plastic_percent",
    "composition_glass_percent",
    "composition_metal_percent",
    "composition_other_percent",
]


@st.cache_resource
def load_waste_model():
    models_dir = ensure_trained("waste", "xgb_waste_model.pkl")
    model = joblib.load(models_dir / "xgb_waste_model.pkl")
    imputer = joblib.load(models_dir / "composition_imputer.pkl")
    with open(models_dir / "feature_columns.json") as f:
        feature_columns = json.load(f)
    return model, imputer, feature_columns


with tab_waste:
    st.header("Municipal waste generation")
    st.caption("Model: XGBoost (R2=0.81), trained on the World Bank What a Waste city-level dataset.")

    model, imputer, feature_columns = load_waste_model()

    population = st.number_input("City population", min_value=1000, value=1_000_000, step=10_000, key="waste_pop")
    income_id = st.selectbox("Income group", INCOME_IDS, key="waste_income")
    region_id = st.selectbox("Region", REGION_IDS, key="waste_region")

    st.caption("Waste composition (%) - leave at 0 to let the model use the dataset median")
    composition_values = {}
    cols = st.columns(3)
    for i, col_name in enumerate(COMPOSITION_COLS):
        label = col_name.replace("composition_", "").replace("_percent", "").replace("_", " ").title()
        composition_values[col_name] = cols[i % 3].number_input(
            label, min_value=0.0, max_value=100.0, value=0.0, key=f"waste_{col_name}"
        )

    if st.button("Predict waste generation"):
        composition_df = pd.DataFrame([composition_values])
        composition_df = composition_df.replace(0.0, np.nan)
        composition_df[:] = imputer.transform(composition_df)

        row = {col: 0.0 for col in feature_columns}
        row["log_population"] = np.log1p(population)
        for col in COMPOSITION_COLS:
            row[col] = composition_df[col].iloc[0]
        income_col = f"income_id_{income_id}"
        if income_col in row:
            row[income_col] = 1.0
        region_col = f"region_id_{region_id}"
        if region_col in row:
            row[region_col] = 1.0

        X = pd.DataFrame([row], columns=feature_columns)
        pred = np.expm1(model.predict(X)[0])
        st.metric("Predicted waste generation", f"{pred:,.0f} tons/year")


# ---------------------------------------------------------------------------
# Traffic
# ---------------------------------------------------------------------------
@st.cache_resource
def load_traffic_model():
    models_dir = ensure_trained("traffic", "rf_traffic_model.pkl")
    model = joblib.load(models_dir / "rf_traffic_model.pkl")
    with open(models_dir / "feature_columns.json") as f:
        feature_columns = json.load(f)
    holidays = sorted({c.split("holiday_", 1)[1] for c in feature_columns if c.startswith("holiday_")})
    weather_main = sorted({c.split("weather_main_", 1)[1] for c in feature_columns if c.startswith("weather_main_")})
    weather_desc = sorted(
        {c.split("weather_description_", 1)[1] for c in feature_columns if c.startswith("weather_description_")}
    )
    return model, feature_columns, holidays, weather_main, weather_desc


with tab_traffic:
    st.header("Traffic volume")
    st.caption("Model: Random Forest (R2=0.96), trained on Metro Interstate Traffic Volume (UCI).")

    t_model, t_feature_columns, t_holidays, t_weather_main, t_weather_desc = load_traffic_model()

    c1, c2 = st.columns(2)
    temp_c = c1.number_input("Temperature (C)", value=20.0, key="traffic_temp")
    clouds_all = c2.slider("Cloud cover %", 0, 100, 40, key="traffic_clouds")
    rain_1h = c1.number_input("Rain in last hour (mm)", min_value=0.0, value=0.0, key="traffic_rain")
    snow_1h = c2.number_input("Snow in last hour (mm)", min_value=0.0, value=0.0, key="traffic_snow")

    c3, c4 = st.columns(2)
    hour = c3.slider("Hour of day", 0, 23, 9, key="traffic_hour")
    dayofweek = c4.selectbox(
        "Day of week", list(range(7)), format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
        key="traffic_dow",
    )
    day = c3.number_input("Day of month", min_value=1, max_value=31, value=15, key="traffic_day")
    month = c4.number_input("Month", min_value=1, max_value=12, value=6, key="traffic_month")

    holiday = st.selectbox("Holiday", ["None"] + t_holidays, key="traffic_holiday")
    weather_main = st.selectbox("Weather", t_weather_main, key="traffic_weather_main")
    weather_description = st.selectbox("Weather detail", t_weather_desc, key="traffic_weather_desc")

    if st.button("Predict traffic volume"):
        row = {col: 0.0 for col in t_feature_columns}
        row["temp"] = temp_c + 273.15  # dataset is in Kelvin
        row["rain_1h"] = rain_1h
        row["snow_1h"] = snow_1h
        row["clouds_all"] = clouds_all
        row["hour"] = hour
        row["day"] = day
        row["month"] = month
        row["dayofweek"] = dayofweek

        if holiday != "None":
            row[f"holiday_{holiday}"] = 1.0
        row[f"weather_main_{weather_main}"] = 1.0
        row[f"weather_description_{weather_description}"] = 1.0

        X = pd.DataFrame([row], columns=t_feature_columns)
        pred = t_model.predict(X)[0]
        st.metric("Predicted traffic volume", f"{pred:,.0f} vehicles/hour")


# ---------------------------------------------------------------------------
# Air Quality
# ---------------------------------------------------------------------------
@st.cache_resource
def load_air_quality_model():
    models_dir = ensure_trained("air_quality", "rf_air_quality_model.pkl")
    model = joblib.load(models_dir / "rf_air_quality_model.pkl")
    with open(models_dir / "feature_columns.json") as f:
        feature_columns = json.load(f)
    return model, feature_columns


def classify_air_quality(co: float) -> str:
    if co <= 2.0:
        return "Good"
    elif co <= 5.0:
        return "Moderate"
    return "Poor"


with tab_air:
    st.header("Air quality (CO concentration)")
    st.caption("Model: Random Forest, trained on the UCI Air Quality sensor dataset.")
    st.caption("Inputs are readings from the other 4 gas sensors on the same monitoring station.")

    a_model, a_feature_columns = load_air_quality_model()

    c1, c2 = st.columns(2)
    pt08_s1 = c1.number_input("PT08.S1 (CO sensor)", value=1100.0, key="air_pt08s1")
    c6h6 = c2.number_input("C6H6(GT) - Benzene", value=10.0, key="air_c6h6")
    pt08_s2 = c1.number_input("PT08.S2 (NMHC sensor)", value=900.0, key="air_pt08s2")
    nox = c2.number_input("NOx(GT)", value=200.0, key="air_nox")
    pt08_s3 = c1.number_input("PT08.S3 (NOx sensor)", value=800.0, key="air_pt08s3")
    no2 = c2.number_input("NO2(GT)", value=110.0, key="air_no2")
    pt08_s4 = c1.number_input("PT08.S4 (NO2 sensor)", value=1500.0, key="air_pt08s4")
    pt08_s5 = c2.number_input("PT08.S5 (O3 sensor)", value=1000.0, key="air_pt08s5")

    c3, c4, c5 = st.columns(3)
    temperature = c3.number_input("Temperature (C)", value=18.0, key="air_temp")
    humidity = c4.number_input("Relative humidity %", value=50.0, key="air_rh")
    abs_humidity = c5.number_input("Absolute humidity", value=1.0, key="air_ah")

    c6, c7, c8, c9 = st.columns(4)
    day = c6.number_input("Day", min_value=1, max_value=31, value=15, key="air_day")
    month = c7.number_input("Month", min_value=1, max_value=12, value=6, key="air_month")
    year = c8.number_input("Year", value=2004, key="air_year")
    hour = c9.slider("Hour", 0, 23, 9, key="air_hour")

    if st.button("Predict air quality"):
        values = {
            "PT08.S1(CO)": pt08_s1,
            "C6H6(GT)": c6h6,
            "PT08.S2(NMHC)": pt08_s2,
            "NOx(GT)": nox,
            "PT08.S3(NOx)": pt08_s3,
            "NO2(GT)": no2,
            "PT08.S4(NO2)": pt08_s4,
            "PT08.S5(O3)": pt08_s5,
            "T": temperature,
            "RH": humidity,
            "AH": abs_humidity,
            "Day": day,
            "Month": month,
            "Year": year,
            "Hour": hour,
        }
        X = pd.DataFrame([values], columns=a_feature_columns)
        pred = float(a_model.predict(X)[0])
        category = classify_air_quality(pred)
        col_a, col_b = st.columns(2)
        col_a.metric("Predicted CO(GT)", f"{pred:.2f} mg/m3")
        col_b.metric("Air quality category", category)


# ---------------------------------------------------------------------------
# Parking
# ---------------------------------------------------------------------------
@st.cache_resource
def load_parking_model():
    models_dir = ensure_trained("parking", "rf_parking_model.pkl")
    model = joblib.load(models_dir / "rf_parking_model.pkl")
    encoder = joblib.load(models_dir / "system_code_encoder.pkl")
    with open(models_dir / "feature_columns.json") as f:
        feature_columns = json.load(f)
    with open(models_dir / "lot_capacity.json") as f:
        lot_capacity = json.load(f)
    return model, encoder, feature_columns, lot_capacity


with tab_parking:
    st.header("Parking occupancy")
    st.caption("Model: Random Forest (R2=0.997), trained on Parking Birmingham (UCI).")
    st.caption("Autoregressive: predicts the next reading from the last 3 known occupancy readings.")

    p_model, p_encoder, p_feature_columns, p_lot_capacity = load_parking_model()

    system_code = st.selectbox("Car park", sorted(p_lot_capacity.keys()), key="parking_lot")
    default_capacity = p_lot_capacity[system_code]
    st.caption(f"Known capacity for this lot: {default_capacity:.0f}")

    c1, c2 = st.columns(2)
    hour = c1.slider("Hour of day", 0, 23, 10, key="parking_hour")
    dayofweek = c2.selectbox(
        "Day of week", list(range(7)), format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
        key="parking_dow",
    )
    month = c1.number_input("Month", min_value=1, max_value=12, value=10, key="parking_month")
    capacity = c2.number_input("Capacity (override)", min_value=1.0, value=float(default_capacity), key="parking_cap")

    st.caption("Last 3 known occupancy readings (most recent first)")
    c3, c4, c5 = st.columns(3)
    lag_1 = c3.number_input("Most recent (lag_1)", min_value=0.0, value=150.0, key="parking_lag1")
    lag_2 = c4.number_input("2nd most recent (lag_2)", min_value=0.0, value=130.0, key="parking_lag2")
    lag_3 = c5.number_input("3rd most recent (lag_3)", min_value=0.0, value=110.0, key="parking_lag3")

    if st.button("Predict next occupancy"):
        encoded = int(p_encoder.transform([system_code])[0])
        row = {
            "Capacity": capacity,
            "Hour": hour,
            "DayOfWeek": dayofweek,
            "Month": month,
            "Weekend": int(dayofweek >= 5),
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "SystemCodeEncoded": encoded,
        }
        X = pd.DataFrame([row], columns=p_feature_columns)
        pred = float(p_model.predict(X)[0])
        col_a, col_b = st.columns(2)
        col_a.metric("Predicted occupancy", f"{pred:,.0f} vehicles")
        col_b.metric("Predicted occupancy rate", f"{pred / capacity * 100:.1f}%")
