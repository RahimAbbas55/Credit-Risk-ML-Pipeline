# Credit Risk ML Pipeline

End-to-end ML pipeline predicting loan default risk using the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset. Built as a proper pipeline — not just a notebook — with data ingestion, multi-table feature engineering, model training, and a served prediction API.

> 🚧 Work in progress — this README will be expanded as the project develops.

## Stack
Python · Pandas · NumPy · Scikit-learn · XGBoost · FastAPI · Docker · AWS

## Progress

**Day 1 — Data Ingestion & EDA**
- Set up project structure, config, and reproducible data loading
- Explored `application_train.csv` (307,511 rows, 122 columns)
- Found class imbalance (~92% / 8% target split)
- Identified two real data quality issues: a placeholder anomaly in `DAYS_EMPLOYED` (365243) and a 247x income outlier

**Day 2 — Feature Engineering (single table)**
- Fixed the `DAYS_EMPLOYED` anomaly (flagged + replaced with NaN)
- Capped income outliers at the 99th percentile
- Handled missing values with column-appropriate strategies (structural fill, median imputation, categorical fill)

**Day 3 — Multi-table Feature Engineering**
- Aggregated `bureau.csv` (1.7M rows) and `previous_application.csv` (1.67M rows) into applicant-level summary features
- Engineered credit history signals (overdue amounts, credit sums, approval/refusal rates)
- Merged everything onto the main table via left joins — zero applicants lost
- Final dataset: 307,511 rows × 137 columns, 0 missing values

**Day 4 — Baseline Model & Evaluation**
- Encoded categorical features (one-hot, 249 total features) and split data with stratification to preserve class balance
- Trained a baseline Logistic Regression model with `class_weight='balanced'` to address the imbalance
- Evaluated using ROC-AUC, precision, recall, and F1 — not accuracy alone, given the class imbalance
- **Results: ROC-AUC 0.754, Recall 0.68 (defaulters), Precision 0.16**
- Saved model, scaler, and evaluation visualizations (ROC curve, confusion matrix) as reusable artifacts
- Consolidated into a standalone `train.py` script — reproducible with one command

**Day 5 — Model Comparison & Selection**
- Trained and compared three models: Logistic Regression, Random Forest, and XGBoost
- Used 5-fold stratified cross-validation for a reliable, unbiased comparison (not just a single train/test split)
- Found Logistic Regression and XGBoost statistically comparable on ROC-AUC (0.751 vs 0.748), with Random Forest clearly behind (0.736) due to a threshold/calibration issue despite similar ranking ability
- Analyzed feature importance: `EXT_SOURCE_1/2/3` (external credit scores) and engineered bureau features dominated XGBoost; Logistic Regression coefficients showed multicollinearity issues (e.g. `AMT_CREDIT` vs `AMT_GOODS_PRICE`)
- **Selected XGBoost as the final model** — comparable accuracy to Logistic Regression, more stable across folds, and more trustworthy feature importances
- Refactored data prep into a single reusable function (`prepare_model_data()`), reducing duplication across notebooks and `train.py`

**Day 6 — Hyperparameter Tuning**
- Tuned XGBoost using `RandomizedSearchCV` with 3-fold cross-validation (20 candidate configurations, 60 total fits)
- Best parameters favored a more regularized configuration: shallower trees (`max_depth=4`), more of them (`n_estimators=300`), lower learning rate (`0.05`), with row/column subsampling (`0.8`) — consistent with the baseline mildly overfitting
- **Results on held-out test set: ROC-AUC 0.757 → 0.767, Recall 0.62 → 0.69** (small precision tradeoff: 0.183 → 0.172)
- Tuning meaningfully improved default detection at a modest cost to false-positive rate — a reasonable tradeoff for a lender prioritizing risk detection
- `train.py` updated with tuned hyperparameters as the final production configuration

**Day 7 — FastAPI Prediction Endpoint & Dockerization**
- Designed a practical API interface: rather than requiring callers to supply all 249 one-hot encoded model features, the API accepts ~15 human-meaningful fields (income, credit amount, age, education, external credit scores, etc.)
- Built a feature defaults template (`feature_defaults.json`) from training data medians/modes, used to fill any fields not provided by the caller
- Implemented `/predict` and `/health` endpoints with Pydantic request/response validation and auto-generated interactive docs
- Verified end-to-end: full-input and minimal-input requests both produce correct, sensible predictions
- Containerized the API with Docker for consistent, portable deployment
- Caught and fixed a real dependency management gap: `fastapi`, `uvicorn`, and `xgboost` were installed locally but missing from `requirements.txt` — only surfaced when tested in Docker's clean environment

**Day 8 — AWS Deployment**
- Pushed the Docker image to AWS ECR (Elastic Container Registry)
- Deployed to a live EC2 instance, configured with proper security groups (SSH restricted to a known IP, API port open publicly)
- Found and fixed a CPU architecture mismatch (image built for arm64, EC2 running amd64) causing `exec format error` crashes — rebuilt using `docker buildx --platform linux/amd64`
- **API is live and publicly accessible**, verified via real HTTP requests from an external machine
- Project complete: full pipeline from raw multi-table data to a live, deployed prediction API

## Live API
- Docs: `http://52.200.3.133:8000/docs`
- Health check: `http://52.200.3.133:8000/health`

> Note: this is a personal EC2 instance for portfolio/demo purposes and may not always be running.

## Setup

```bash
git clone <repo-url>
cd Credit-Risk-ML-Pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Data isn't included (size + licensing). Download from the [Kaggle competition page](https://www.kaggle.com/c/home-credit-default-risk) and place CSVs in `data/raw/`.

## Training

```bash
python -m src.train
```

## Running the API

```bash
uvicorn api.main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

Or via Docker:
```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

## Possible future improvements
- CI/CD via GitHub Actions (automated testing, linting)
- Elastic IP for a stable, permanent address
- HTTPS via a reverse proxy (e.g. Nginx + Let's Encrypt)