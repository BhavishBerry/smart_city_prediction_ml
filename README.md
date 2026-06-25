# Smart City Prediction (ML/DL)

Summer internship group project: predicting key smart-city signals using classical ML and deep learning, served through a shared API + dashboard.

## Team domains

| Domain | Folder | Owner | Dataset | Deployed model | R² |
|---|---|---|---|---|---|
| Traffic flow | `traffic/` | Anshika | Metro Interstate Traffic Volume (UCI) | Random Forest | 0.96 |
| Air quality / AQI | `air_quality/` | Rishu | Air Quality (UCI) | Random Forest | 0.939 |
| Parking occupancy | `parking/` | Navish | Parking Birmingham (UCI) | Random Forest | 0.997 |
| Waste generation | `waste/` | Bhavish | What a Waste Global Database (World Bank) | XGBoost | 0.81 (CV mean 0.78) |

See [ALGORITHMS.md](ALGORITHMS.md) for the full reasoning behind each dataset and algorithm choice.

All datasets are real, publicly published data — no synthetic or AI-generated data is used anywhere in this repository.

---

## Project structure

```
smart_city_prediction_ml/
├── traffic/
│   ├── data/
│   │   └── Metro_Interstate_Traffic_Volume.csv
│   ├── notebooks/          # EDA + experimentation (pending: Anshika's notebooks)
│   ├── src/
│   │   └── train_model.py  # trains RF, saves rf_traffic_model.pkl
│   └── models/             # gitignored .pkl artifacts (auto-generated on first run)
│
├── air_quality/
│   ├── data/
│   │   └── AirQualityUCI.csv
│   ├── notebooks/
│   │   └── ttest.ipynb     # statistical analysis (in progress)
│   ├── src/
│   │   └── train_model.py  # trains RF on CO(GT) target, saves rf_air_quality_model.pkl
│   └── models/
│
├── parking/
│   ├── data/
│   │   └── Parking_Birmingham.csv
│   ├── notebooks/          # pending: Navish's EDA + baseline notebook
│   ├── src/
│   │   └── train_model.py  # trains autoregressive RF, saves rf_parking_model.pkl
│   └── models/
│
├── waste/
│   ├── data/
│   │   ├── WhatAWaste_City_Level.csv
│   │   ├── WhatAWaste_City_Level_Codebook.csv
│   │   ├── WhatAWaste_Country_Level.csv
│   │   └── WhatAWaste_Country_Level_Codebook.csv
│   ├── notebooks/
│   │   ├── data_exploration.ipynb   # EDA — feature selection, correlation analysis
│   │   ├── 00_basemodel.ipynb       # ML baseline: RF + XGBoost with 5-fold CV
│   │   └── 01_deep_learning.ipynb   # DL comparison: MLP (underperforms XGBoost here)
│   ├── src/
│   │   └── train_model.py  # trains XGBoost (best model), saves xgb_waste_model.pkl
│   └── models/
│
├── shared/
│   ├── api/
│   │   └── main.py         # FastAPI app — all 4 /predict endpoints
│   └── dashboard/
│       └── app.py          # Streamlit dashboard — live monitoring + what-if modes
│
├── .streamlit/
│   └── config.toml         # light creamish theme
├── .github/
│   └── workflows/
│       └── branch-up-to-date.yml  # CI: fails if a branch falls behind main
├── requirements.txt
├── ALGORITHMS.md
└── CONTRIBUTING.md
```

---

## Datasets

| Domain | File(s) | Source | Notes |
|---|---|---|---|
| Traffic | `traffic/data/Metro_Interstate_Traffic_Volume.csv` | [UCI #492](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) | Hourly I-94 traffic + weather, 2012–2018 |
| Air quality | `air_quality/data/AirQualityUCI.csv` | [UCI #360](https://archive.ics.uci.edu/dataset/360/air+quality) | Hourly sensor readings, semicolon-delimited, decimal commas, `-200` = missing |
| Parking | `parking/data/Parking_Birmingham.csv` | [UCI #482](https://archive.ics.uci.edu/dataset/482/parking+birmingham) | ~15-30 min intervals, 30+ Birmingham car parks, Oct–Dec 2016 |
| Waste | `waste/data/WhatAWaste_City_Level.csv` + country-level + codebooks | [World Bank](https://datacatalog.worldbank.org/search/dataset/0039597/what-a-waste-global-database) | Cross-sectional (one row per city/country), not a time series |

Raw datasets are committed directly (no `data/raw/` exclusion at this scale). Model `.pkl` files are gitignored and regenerated automatically.

---

## Notebooks

### Waste (Bhavish) — complete
| Notebook | Purpose |
|---|---|
| `waste/notebooks/data_exploration.ipynb` | EDA: feature correlation, population dominance finding (r=0.89), target/population skew analysis |
| `waste/notebooks/00_basemodel.ipynb` | ML baseline: Random Forest + XGBoost with log-transform, one-hot encoding, 5-fold CV for honest R² estimate; XGBoost wins (R²=0.81 single split, CV mean 0.78) |
| `waste/notebooks/01_deep_learning.ipynb` | DL comparison: MLP on same features; underperforms XGBoost on this cross-sectional dataset |

### Air quality (Rishu) — in progress
| Notebook | Purpose |
|---|---|
| `air_quality/notebooks/ttest.ipynb` | Statistical analysis (in progress) |
| `air_quality/notebooks/00_basemodel.ipynb` | Pending — RF baseline (R²=0.939 on original run) |
| `air_quality/notebooks/01_deep_learning.ipynb` | Pending — LSTM / CNN-LSTM |

### Traffic (Anshika) — pending
Notebooks not yet pushed. Planned: EDA + RF/XGBoost baseline (R²=0.96) + LSTM comparison.

### Parking (Navish) — pending
Notebooks not yet pushed. Planned: EDA + RF baseline (R²=0.997) + LSTM comparison.

---

## Models

Each domain's `src/train_model.py` mirrors the best model from the notebooks and is the single source of truth for what gets deployed. Model artifacts are gitignored (binary, regenerable):

| File | Domain | Algorithm | Notes |
|---|---|---|---|
| `traffic/models/rf_traffic_model.pkl` | Traffic | Random Forest | 200 estimators, max_depth=15 |
| `air_quality/models/rf_air_quality_model.pkl` | Air quality | Random Forest | Predicts CO(GT); unscaled (RF doesn't need scaling) |
| `parking/models/rf_parking_model.pkl` + `system_code_encoder.pkl` + `lot_capacity.json` | Parking | Random Forest | Autoregressive next-reading model; lot-aware capacity lookup |
| `waste/models/xgb_waste_model.pkl` + `composition_imputer.pkl` | Waste | XGBoost | Log-transformed target; population is dominant feature |

All models are **auto-trained on first run** if the `.pkl` is missing — a fresh clone requires no manual training step.

---

## Setup

```bash
git clone https://github.com/BhavishBerry/smart_city_prediction_ml.git
cd smart_city_prediction_ml
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Dependencies:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow`, `matplotlib`, `seaborn`, `jupyter`, `fastapi`, `uvicorn`, `streamlit`, `streamlit-autorefresh`, `joblib`, `requests`

---

## Running the app

Run each in a separate terminal from the project root:

**Backend (FastAPI):**
```bash
uvicorn shared.api.main:app --reload --port 8000
```
- API: `http://localhost:8000`
- Interactive docs (Swagger UI): `http://localhost:8000/docs`

**Frontend (Streamlit dashboard):**
```bash
streamlit run shared/dashboard/app.py
```
- Dashboard: `http://localhost:8501`

Both services auto-train missing models on first startup — first run may take a minute.

---

## API endpoints

All endpoints accept `POST` with a JSON body. Categorical values outside the trained vocabulary return HTTP 422 with a descriptive error.

| Endpoint | Predicts |
|---|---|
| `POST /predict/traffic` | Hourly traffic volume |
| `POST /predict/air_quality` | CO(GT) pollutant concentration |
| `POST /predict/parking` | Next occupancy reading for a car park |
| `POST /predict/waste` | Annual municipal solid waste generation (tons/year) |

---

## Dashboard

Two modes, switchable from the sidebar:

**Live City Monitoring** (default)
- All 4 domains shown side by side as status cards
- Each card: predicted value, qualitative severity tier (Low / Moderate / High, color-coded), rolling trend chart, recommended action when a threshold is crossed
- Status banner at top: "All systems normal" vs count of systems needing attention
- Feed replays real historical rows with the timestamp overridden to now — clearly labeled as a simulated feed (no live sensor connection)
- Auto-refreshes every 20 s by default (adjustable 10–60 s in sidebar)
- Each card shows the model algorithm, its R², and a short operational explainer

**Manual What-If**
- Per-domain input forms to test a specific scenario manually
- Returns the raw model prediction for that input

Sidebar also has an "About the models" reference summarising all 4 models' value propositions.

---

## CI

`.github/workflows/branch-up-to-date.yml` runs on every push and PR and fails if the branch is behind `main`. Fix with:

```bash
git fetch origin && git merge origin/main && git push
```

---

## Workflow

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full branch workflow (naming conventions, daily flow, conflict rules, PR process).

---

## Pending

- Anshika: push traffic EDA + baseline notebook (`traffic/notebooks/`)
- Rishu: complete air quality baseline notebook (`air_quality/notebooks/00_basemodel.ipynb`) and DL notebook
- Navish: push parking EDA + baseline notebook (`parking/notebooks/`)
- All: clear notebook outputs before committing (`Kernel > Restart & Clear Output`)
