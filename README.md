# Smart City Prediction (ML/DL)

Summer internship group project: predicting key smart-city signals using classical ML and deep learning, served through a shared API + dashboard.

## Team domains

| Domain | Folder | Owner |
|---|---|---|
| Traffic flow prediction (simple) | `traffic/` | TBD |
| Air quality / AQI prediction (simple) | `air_quality/` | TBD |
| Parking occupancy prediction | `parking/` | TBD |
| Waste generation prediction | `waste/` | Bhavish |

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
