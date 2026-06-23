# Smart City Prediction (ML/DL)

Summer internship group project: predicting key smart-city signals using classical ML and deep learning, served through a shared API + dashboard.

## Team domains

| Domain | Folder | Owner | Dataset | Algorithms |
|---|---|---|---|---|
| Traffic flow prediction | `traffic/` | Anshika | [Metro Interstate Traffic Volume (UCI)](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) | XGBoost/Random Forest (ML) + LSTM (DL) |
| Air quality / AQI prediction | `air_quality/` | Rishu | [Air Quality (UCI)](https://archive.ics.uci.edu/dataset/360/air+quality) | Random Forest (ML) + LSTM/CNN-LSTM (DL) |
| Parking occupancy prediction | `parking/` | Navish | [Parking Birmingham (UCI)](https://archive.ics.uci.edu/dataset/482/parking+birmingham) | Random Forest (primary) + LSTM (comparison) |
| Waste generation prediction | `waste/` | Bhavish | [What a Waste Global Database (World Bank)](https://datacatalog.worldbank.org/search/dataset/0039597/what-a-waste-global-database) | Random Forest/XGBoost (ML) + MLP (DL) |

See [ALGORITHMS.md](ALGORITHMS.md) for the full reasoning behind each dataset and algorithm choice.

All datasets are real, publicly published data — no synthetic or AI-generated data is used anywhere in this repository.

## Structure

Each domain folder is self-contained:

```
<domain>/
  data/        # raw + processed datasets (gitignored except small samples)
  notebooks/   # EDA and experimentation
  src/         # data generation/loading, training, inference scripts
  models/      # saved model artifacts (gitignored)
```

`shared/` holds cross-domain pieces:

```
shared/
  api/         # FastAPI app exposing /predict endpoints per domain
  dashboard/   # Streamlit dashboard visualizing all 4 domains
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Workflow

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch workflow each team member follows.
