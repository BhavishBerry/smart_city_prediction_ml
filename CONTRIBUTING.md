# Branch workflow

Repo: `https://github.com/BhavishBerry/smart_city_prediction_ml.git`

`main` is protected by convention — nobody pushes to it directly except via PR. Each person works in their own domain folder on their own branch.

## One-time setup (each teammate, on their own laptop)

```bash
git clone https://github.com/BhavishBerry/smart_city_prediction_ml.git
cd smart_city_prediction_ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Branch naming

```
feature/<domain>-<yourname>
```

Assigned branches:

- `feature/waste-prediction-bhavish` — Bhavish
- `feature/traffic-prediction-anshika` — Anshika
- `feature/air-quality-prediction-rishu` — Rishu
- `feature/parking-prediction-navish` — Navish

## Daily flow

```bash
git checkout main
git pull origin main
git checkout -b feature/<domain>-<yourname>   # first time only
# ... work inside your domain folder only (traffic/, air_quality/, parking/, or waste/) ...
git add <your-domain-folder>
git commit -m "describe what changed"
git push -u origin feature/<domain>-<yourname>
```

Then open a Pull Request on GitHub into `main`. Whoever isn't the author gives it a quick look before merging — this keeps everyone's `main` consistent and avoids merge conflicts since each person only touches their own folder.

## Rules to avoid conflicts

- Only edit files inside **your own domain folder**. If you need something in `shared/` (API or dashboard), say so in the PR description so it's reviewed together.
- Pull `main` before starting work each session: `git checkout main && git pull`.
- Keep notebooks' outputs cleared before committing (`Kernel > Restart & Clear Output`) to avoid noisy diffs.
- Don't commit raw datasets or model binaries — `.gitignore` already excludes `data/raw/`, `models/*.pkl`, `*.h5`, etc. Add a small `data/sample.csv` if teammates need something to test against.

## Once all 4 domains have a baseline model

We'll integrate everyone's `src/predict.py` into `shared/api/main.py` as separate endpoints (`/predict/traffic`, `/predict/air_quality`, `/predict/parking`, `/predict/waste`), and wire `shared/dashboard/app.py` to call all four.
