"""Smart city prediction dashboard.

Two modes:
- Live City Monitoring (default): simulates a live sensor feed by replaying
  randomized real historical rows from each dataset (NOT synthetic data -
  the underlying values are real, just resampled with the clock overridden
  to "now" to simulate a live read). Clearly labeled as simulated in the UI.
- Manual What-If: the original per-domain input forms for testing a single
  specific scenario.

Loads each domain's model directly (no API dependency). Model artifacts are
gitignored, so every loader auto-trains on first run if its .pkl is missing.
"""

import importlib.util
import json
import random
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

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


st.set_page_config(page_title="Smart City Predictions", layout="wide", page_icon="\U0001F3D9️")

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


# ---------------------------------------------------------------------------
# Model loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_waste_model():
    models_dir = ensure_trained("waste", "xgb_waste_model.pkl")
    model = joblib.load(models_dir / "xgb_waste_model.pkl")
    imputer = joblib.load(models_dir / "composition_imputer.pkl")
    with open(models_dir / "feature_columns.json") as f:
        feature_columns = json.load(f)
    return model, imputer, feature_columns


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


@st.cache_resource
def load_air_quality_model():
    models_dir = ensure_trained("air_quality", "rf_air_quality_model.pkl")
    model = joblib.load(models_dir / "rf_air_quality_model.pkl")
    with open(models_dir / "feature_columns.json") as f:
        feature_columns = json.load(f)
    return model, feature_columns


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


def classify_air_quality(co: float) -> str:
    if co <= 2.0:
        return "Good"
    elif co <= 5.0:
        return "Moderate"
    return "Poor"


# ---------------------------------------------------------------------------
# Raw dataset pools, for resampling real rows into a simulated live feed
# ---------------------------------------------------------------------------
@st.cache_data
def load_traffic_pool():
    df = pd.read_csv(REPO_ROOT / "traffic" / "data" / "Metro_Interstate_Traffic_Volume.csv")
    df = df.drop_duplicates()
    low_q, high_q = df["traffic_volume"].quantile([1 / 3, 2 / 3])
    p05, p95 = df["traffic_volume"].quantile([0.05, 0.95])
    return df, float(low_q), float(high_q), float(p05), float(p95)


@st.cache_data
def load_air_quality_pool():
    df = pd.read_csv(REPO_ROOT / "air_quality" / "data" / "AirQualityUCI.csv")
    df = df.drop(columns=["NMHC(GT)"]).dropna()
    return df


@st.cache_data
def load_parking_pool():
    df = pd.read_csv(REPO_ROOT / "parking" / "data" / "Parking_Birmingham.csv")
    df["LastUpdated"] = pd.to_datetime(df["LastUpdated"])
    df = df.sort_values(["SystemCodeNumber", "LastUpdated"]).reset_index(drop=True)
    return df


@st.cache_data
def load_waste_pool():
    df = pd.read_csv(REPO_ROOT / "waste" / "data" / "WhatAWaste_City_Level.csv")
    target = "total_msw_total_msw_generated_tons_year"
    df[target] = pd.to_numeric(df[target].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df = df.dropna(subset=[target])
    income_group_means = df.groupby("income_id")[target].mean().to_dict()
    p95 = float(df[target].quantile(0.95))
    return df, income_group_means, p95


# ---------------------------------------------------------------------------
# Live-feed sample generators - resample a real row, override the clock to "now"
# ---------------------------------------------------------------------------
def sample_traffic_reading(model, feature_columns, pool, now):
    raw = pool.sample(1).iloc[0]
    row = {col: 0.0 for col in feature_columns}
    row["temp"] = raw["temp"]
    row["rain_1h"] = raw["rain_1h"]
    row["snow_1h"] = raw["snow_1h"]
    row["clouds_all"] = raw["clouds_all"]
    row["hour"] = now.hour
    row["day"] = now.day
    row["month"] = now.month
    row["dayofweek"] = now.weekday()
    if pd.notna(raw["holiday"]):
        col = f"holiday_{raw['holiday']}"
        if col in row:
            row[col] = 1.0
    wm_col = f"weather_main_{raw['weather_main']}"
    if wm_col in row:
        row[wm_col] = 1.0
    wd_col = f"weather_description_{raw['weather_description']}"
    if wd_col in row:
        row[wd_col] = 1.0
    X = pd.DataFrame([row], columns=feature_columns)
    pred = float(model.predict(X)[0])
    return {
        "value": pred,
        "detail": f"{raw['weather_main']} ({raw['weather_description']}), {raw['temp'] - 273.15:.1f}C",
    }


def sample_air_quality_reading(model, feature_columns, pool, now):
    raw = pool.sample(1).iloc[0]
    values = {col: raw[col] for col in feature_columns if col not in ("Day", "Month", "Year", "Hour")}
    values["Day"] = now.day
    values["Month"] = now.month
    values["Year"] = now.year
    values["Hour"] = now.hour
    X = pd.DataFrame([values], columns=feature_columns)
    pred = float(model.predict(X)[0])
    return {"value": pred, "detail": f"benzene {raw['C6H6(GT)']:.1f}, NOx {raw['NOx(GT)']:.0f}"}


def sample_parking_reading(model, encoder, feature_columns, pool, now):
    lot = random.choice(pool["SystemCodeNumber"].unique().tolist())
    lot_rows = pool[pool["SystemCodeNumber"] == lot].reset_index(drop=True)
    if len(lot_rows) < 4:
        idx = 0
    else:
        idx = random.randint(3, len(lot_rows) - 1)
    window = lot_rows.iloc[idx - 3 : idx]
    capacity = float(lot_rows["Capacity"].iloc[0])
    encoded = int(encoder.transform([lot])[0])
    row = {
        "Capacity": capacity,
        "Hour": now.hour,
        "DayOfWeek": now.weekday(),
        "Month": now.month,
        "Weekend": int(now.weekday() >= 5),
        "lag_1": float(window["Occupancy"].iloc[-1]),
        "lag_2": float(window["Occupancy"].iloc[-2]),
        "lag_3": float(window["Occupancy"].iloc[-3]),
        "SystemCodeEncoded": encoded,
    }
    X = pd.DataFrame([row], columns=feature_columns)
    pred = float(model.predict(X)[0])
    return {"value": pred, "rate": pred / capacity if capacity else 0.0, "detail": f"{lot} (capacity {capacity:.0f})"}


def sample_waste_reading(model, imputer, feature_columns, pool):
    raw = pool.sample(1).iloc[0]
    composition = pd.DataFrame([[raw[c] for c in COMPOSITION_COLS]], columns=COMPOSITION_COLS)
    composition[:] = imputer.transform(composition)

    row = {col: 0.0 for col in feature_columns}
    row["log_population"] = np.log1p(raw["population_population_number_of_people"])
    for col in COMPOSITION_COLS:
        row[col] = composition[col].iloc[0]
    income_col = f"income_id_{raw['income_id']}"
    if income_col in row:
        row[income_col] = 1.0
    region_col = f"region_id_{raw['region_id']}"
    if region_col in row:
        row[region_col] = 1.0

    X = pd.DataFrame([row], columns=feature_columns)
    pred = float(np.expm1(model.predict(X)[0]))
    return {
        "value": pred,
        "income_id": raw["income_id"],
        "detail": f"{raw['city_name']}, {raw['country_name']} ({raw['income_id']})",
    }


STATUS_COLORS = {"Low": "🟢", "Good": "🟢", "Moderate": "🟡", "High": "🔴", "Poor": "🔴", "Near Full": "🔴"}
STATUS_OK = {"Low", "Good"}
STATUS_SEVERE = {"High", "Poor", "Near Full"}

MODEL_INFO = {
    "traffic": {
        "algorithm": "Random Forest Regressor",
        "score": "R² = 0.96",
        "value": "Forecasts hourly traffic volume so signal timing and rerouting can be adjusted "
        "ahead of congestion building, instead of reacting after gridlock has already formed.",
    },
    "air": {
        "algorithm": "Random Forest Regressor",
        "score": "R² = 0.94",
        "value": "Predicts CO concentration from co-located sensor readings, catching pollution "
        "spikes early enough to issue health advisories before air quality turns unsafe.",
    },
    "parking": {
        "algorithm": "Random Forest Regressor (autoregressive)",
        "score": "R² = 0.997",
        "value": "Predicts each car park's next occupancy reading from its recent trend, so drivers "
        "can be guided to open lots before they arrive and start circling for a spot.",
    },
    "waste": {
        "algorithm": "XGBoost Regressor",
        "score": "R² = 0.81 (5-fold CV: 0.78)",
        "value": "Forecasts annual waste generation per city, so collection routes and truck "
        "dispatch can be planned ahead of overflow rather than after residents complain.",
    },
}

ACTIONS = {
    ("traffic", "Low"): "Traffic flowing normally — no action needed.",
    ("traffic", "Moderate"): "Traffic building. Keep monitoring for further increases.",
    ("traffic", "High"): "⚠ Recommended action: extend green-light duration on main corridors and "
    "push a reroute alert to the city traffic app.",
    ("air", "Good"): "Air quality safe — no action needed.",
    ("air", "Moderate"): "Air quality declining. Watch sensitive areas (schools, hospitals).",
    ("air", "Poor"): "⚠ Recommended action: issue a public health advisory and consider temporary "
    "heavy-vehicle restrictions.",
    ("parking", "Low"): "Ample space available — no action needed.",
    ("parking", "Moderate"): "Lot filling up. Keep monitoring.",
    ("parking", "Near Full"): "⚠ Recommended action: redirect incoming traffic to nearby lots with "
    "availability via signage/app.",
    ("waste", "Low"): "Generation within normal range for this income bracket.",
    ("waste", "Moderate"): "Slightly above average for this income bracket. Keep monitoring.",
    ("waste", "High"): "⚠ Recommended action: schedule an additional collection run for this zone "
    "this week.",
}


def render_card(col, domain, title, icon, reading, status, history_key):
    with col:
        with st.container(border=True):
            info = MODEL_INFO[domain]
            st.markdown(f"#### {icon} {title}")
            st.caption(f"{info['algorithm']} • {info['score']}")

            badge = STATUS_COLORS.get(status, "⚪")
            st.markdown(f"**Status: {badge} {status}**")
            st.metric("Predicted value", reading["display"])

            gauge = max(0.0, min(1.0, reading.get("gauge", 0.0)))
            st.progress(gauge)

            action = ACTIONS.get((domain, status), "")
            if status in STATUS_SEVERE:
                st.warning(action, icon="⚠️")
            elif status in STATUS_OK:
                st.caption(action)
            else:
                st.info(action, icon="👀")

            st.caption(reading.get("detail", ""))
            st.caption(f"Simulated reading as of {datetime.now().strftime('%H:%M:%S')}")

            history = st.session_state.history[history_key]
            if len(history) > 1:
                st.line_chart(pd.DataFrame(history, columns=["value"]), height=120)

            with st.expander("Why this model?"):
                st.write(info["value"])


# ---------------------------------------------------------------------------
# Sidebar / mode selection
# ---------------------------------------------------------------------------
st.title("\U0001F3D9️ Smart City Maintenance Dashboard")
st.caption("Predictive monitoring across traffic, air quality, parking, and waste — built to catch problems before they happen, not just report them after.")

mode = st.sidebar.radio("Mode", ["Live City Monitoring", "Manual What-If"])

if mode == "Live City Monitoring":
    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    interval = st.sidebar.slider("Refresh interval (seconds)", 10, 60, 20, step=5)
    force_refresh = st.sidebar.button("Refresh now")

    with st.sidebar.expander("About the models", expanded=False):
        for domain, info in MODEL_INFO.items():
            st.markdown(f"**{domain.replace('_', ' ').title()}** — {info['algorithm']}, {info['score']}")
            st.caption(info["value"])

    st.info(
        "**Simulated live feed** - readings below are randomly resampled from the real historical "
        "datasets (not synthetic/invented data) with the clock overridden to the current time, since "
        "no live sensor feed is connected. This is for demo purposes only.",
        icon="ℹ️",
    )

    waste_model, waste_imputer, waste_feature_columns = load_waste_model()
    traffic_model, traffic_feature_columns, *_ = load_traffic_model()
    air_model, air_feature_columns = load_air_quality_model()
    parking_model, parking_encoder, parking_feature_columns, parking_lot_capacity = load_parking_model()

    traffic_pool, traffic_low_q, traffic_high_q, traffic_p05, traffic_p95 = load_traffic_pool()
    air_pool = load_air_quality_pool()
    parking_pool = load_parking_pool()
    waste_pool, waste_income_means, waste_p95 = load_waste_pool()

    if auto_refresh:
        count = st_autorefresh(interval=interval * 1000, key="live_counter")
    else:
        count = 0

    if "history" not in st.session_state:
        st.session_state.history = {"traffic": [], "air": [], "parking": [], "waste": []}
        st.session_state.last_count = -1

    should_sample = force_refresh or count != st.session_state.last_count or not st.session_state.get("current")
    if should_sample:
        st.session_state.last_count = count
        now = datetime.now()

        traffic_reading = sample_traffic_reading(traffic_model, traffic_feature_columns, traffic_pool, now)
        if traffic_reading["value"] < traffic_low_q:
            traffic_status = "Low"
        elif traffic_reading["value"] < traffic_high_q:
            traffic_status = "Moderate"
        else:
            traffic_status = "High"
        traffic_reading["display"] = f"{traffic_reading['value']:,.0f} vehicles/hr"
        traffic_reading["gauge"] = (traffic_reading["value"] - traffic_p05) / (traffic_p95 - traffic_p05)

        air_reading = sample_air_quality_reading(air_model, air_feature_columns, air_pool, now)
        air_status = classify_air_quality(air_reading["value"])
        air_reading["display"] = f"{air_reading['value']:.2f} mg/m3 CO"
        air_reading["gauge"] = air_reading["value"] / 10.0

        parking_reading = sample_parking_reading(parking_model, parking_encoder, parking_feature_columns, parking_pool, now)
        rate = parking_reading["rate"]
        parking_status = "Near Full" if rate > 0.85 else ("Moderate" if rate > 0.5 else "Low")
        parking_reading["display"] = f"{rate * 100:.0f}% full"
        parking_reading["gauge"] = rate

        waste_reading = sample_waste_reading(waste_model, waste_imputer, waste_feature_columns, waste_pool)
        group_mean = waste_income_means.get(waste_reading["income_id"], waste_reading["value"])
        waste_status = "High" if waste_reading["value"] > group_mean * 1.2 else "Moderate" if waste_reading["value"] > group_mean * 0.8 else "Low"
        waste_reading["display"] = f"{waste_reading['value']:,.0f} tons/yr"
        waste_reading["gauge"] = waste_reading["value"] / waste_p95

        st.session_state.current = {
            "traffic": (traffic_reading, traffic_status),
            "air": (air_reading, air_status),
            "parking": (parking_reading, parking_status),
            "waste": (waste_reading, waste_status),
        }
        for key in ("traffic", "air", "parking", "waste"):
            st.session_state.history[key].append(st.session_state.current[key][0]["value"])
            st.session_state.history[key] = st.session_state.history[key][-30:]

    statuses = [v[1] for v in st.session_state.current.values()]
    alert_count = sum(1 for s in statuses if s in STATUS_SEVERE)
    if alert_count == 0:
        st.success(f"All systems normal as of {datetime.now().strftime('%H:%M:%S')}.", icon="✅")
    else:
        st.warning(f"{alert_count} system(s) need attention as of {datetime.now().strftime('%H:%M:%S')}.", icon="🚨")

    c1, c2, c3, c4 = st.columns(4)
    render_card(c1, "traffic", "Traffic", "\U0001F697", *st.session_state.current["traffic"], "traffic")
    render_card(c2, "air", "Air Quality", "\U0001F32B️", *st.session_state.current["air"], "air")
    render_card(c3, "parking", "Parking", "\U0001F17F️", *st.session_state.current["parking"], "parking")
    render_card(c4, "waste", "Waste", "\U0001F5D1️", *st.session_state.current["waste"], "waste")

else:
    tab_waste, tab_traffic, tab_air, tab_parking = st.tabs(["Waste", "Traffic", "Air Quality", "Parking"])

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
            row["temp"] = temp_c + 273.15
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
