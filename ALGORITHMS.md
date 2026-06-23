# Datasets and algorithm choices

All four datasets below are real, publicly published data (UCI Machine Learning Repository, World Bank). No synthetic or generated data is used in this repository.

## Traffic flow prediction — Anshika

- **Dataset**: [Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) (UCI). Hourly westbound I-94 traffic volume near Minneapolis-St Paul, 2012–2018, with hourly weather (temp, rain, snow, cloud cover) and US holidays.
- **File**: `traffic/data/Metro_Interstate_Traffic_Volume.csv`
- **Structure**: hourly time series.
- **Target**: `traffic_volume`.
- **ML baseline**: XGBoost / Random Forest regression on lag + calendar + weather features — consistently among the best non-deep methods for traffic volume in the literature.
- **DL model**: LSTM (or GRU) over a sliding window of recent hours — captures the temporal dependencies that tree models miss. A CNN-LSTM hybrid is the documented best performer if there's time to go further.

## Air quality / AQI prediction — Rishu

- **Dataset**: [Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality) (UCI). Hourly gas multisensor device readings (CO, NMHC, NOx, NO2, benzene) in a polluted Italian city, March 2004–February 2005.
- **File**: `air_quality/data/AirQualityUCI.csv`
- **Structure**: hourly time series, semicolon-delimited, decimal commas, `-200` marks missing sensor readings (needs cleaning before modeling).
- **Target**: pick one pollutant concentration (e.g. `CO(GT)` or `C6H6(GT)`) or derive a composite AQI-style score.
- **ML baseline**: Random Forest regression — repeatedly the strongest shallow-learning model for this kind of sensor data.
- **DL model**: LSTM, ideally CNN-LSTM — literature shows DL outperforming shallow ML for AQI specifically because of strong temporal/seasonal dependence.

## Parking occupancy prediction — Navish

- **Dataset**: [Parking Birmingham](https://archive.ics.uci.edu/dataset/482/parking+birmingham) (UCI). Occupancy and capacity for 30+ car parks in Birmingham, UK, Oct–Dec 2016, ~15-30 min intervals.
- **File**: `parking/data/Parking_Birmingham.csv`
- **Structure**: time series per car park (`SystemCodeNumber`).
- **Target**: `Occupancy` (or `Occupancy / Capacity` as an occupancy rate).
- **ML baseline / primary model**: Random Forest. Comparative studies on this exact problem found simpler models (Random Forest, Decision Tree, KNN) beat more complex ones (MLP) on accuracy — so RF is the primary model here, not just a baseline.
- **DL model (optional/comparison)**: LSTM over each car park's occupancy sequence, mainly to confirm/contrast against the RF result rather than to chase higher accuracy.

## Waste generation prediction — Bhavish

- **Dataset**: [What a Waste Global Database](https://datacatalog.worldbank.org/search/dataset/0039597/what-a-waste-global-database) (World Bank). City-level and country-level solid waste generation, composition, collection coverage, and management data for 200+ countries / 300+ cities.
- **Files**: `waste/data/WhatAWaste_City_Level.csv`, `waste/data/WhatAWaste_Country_Level.csv` (+ codebooks describing every column).
- **Structure**: **cross-sectional**, not time series — one row per city/country, not per date. This is the key difference from the other three domains and changes the algorithm choice.
- **Target**: `total_msw_total_msw_generated_tons_year` (total municipal solid waste generated per year), predicted from population, income group, region, and waste composition features.
- **ML baseline**: Random Forest / XGBoost / Gradient Boosting — this is exactly what recent papers on this same World Bank dataset use, since there's no temporal axis to exploit.
- **DL model**: a feedforward neural network (MLP), not LSTM — there's no sequence/time dimension in this dataset, so a recurrent model would add complexity without benefit. The DL model here is about learning nonlinear interactions between population, income, and composition features, not temporal patterns.

## Why these picks

Algorithm choice was driven by a literature/best-practices pass per domain (UCI/Kaggle dataset search + papers comparing models on these problems), not by familiarity or convenience — in particular, waste uses an MLP instead of an LSTM specifically because its data has no time axis, unlike the other three domains.
