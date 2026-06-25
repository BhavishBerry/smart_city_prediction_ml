"""Smart city prediction dashboard - one tab per domain.

Only the Waste tab is fully wired up (loads the model directly rather than
calling the API, so the dashboard works standalone without the API running).
The other three are placeholders until their owners add a trained model.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WASTE_MODELS_DIR = REPO_ROOT / "waste" / "models"

st.set_page_config(page_title="Smart City Predictions", layout="centered")
st.title("Smart City Predictions")

tab_waste, tab_traffic, tab_air, tab_parking = st.tabs(
    ["Waste", "Traffic", "Air Quality", "Parking"]
)

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
    if not (WASTE_MODELS_DIR / "xgb_waste_model.pkl").exists():
        import sys

        sys.path.insert(0, str(REPO_ROOT / "waste" / "src"))
        from train_model import main as train_waste_model

        train_waste_model()

    model = joblib.load(WASTE_MODELS_DIR / "xgb_waste_model.pkl")
    imputer = joblib.load(WASTE_MODELS_DIR / "composition_imputer.pkl")
    with open(WASTE_MODELS_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)
    return model, imputer, feature_columns


with tab_waste:
    st.header("Municipal waste generation")
    st.caption("Model: XGBoost, trained on the World Bank What a Waste city-level dataset.")

    model, imputer, feature_columns = load_waste_model()

    population = st.number_input("City population", min_value=1000, value=1_000_000, step=10_000)
    income_id = st.selectbox("Income group", INCOME_IDS)
    region_id = st.selectbox("Region", REGION_IDS)

    st.caption("Waste composition (%) - leave at 0 to let the model use the dataset median")
    composition_values = {}
    cols = st.columns(3)
    for i, col_name in enumerate(COMPOSITION_COLS):
        label = col_name.replace("composition_", "").replace("_percent", "").replace("_", " ").title()
        composition_values[col_name] = cols[i % 3].number_input(label, min_value=0.0, max_value=100.0, value=0.0)

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
    st.header("Traffic flow")
    st.info("Coming soon - waiting on a saved model from traffic/notebooks/00_basemodel.ipynb.")

with tab_air:
    st.header("Air quality")
    st.info("Coming soon - waiting on a saved model from air_quality/notebooks/00_basemodel.ipynb.")

with tab_parking:
    st.header("Parking occupancy")
    st.info("Coming soon - waiting on a saved model from parking/notebooks/00_basemodel.ipynb.")
