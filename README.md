# Credit Risk ML Pipeline

End-to-end ML pipeline predicting loan default risk using the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset. Built as a proper pipeline — not just a notebook — with data ingestion, multi-table feature engineering, model training, and (soon) a served prediction API.

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

## Setup

```bash
git clone <repo-url>
cd Credit-Risk-ML-Pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Data isn't included (size + licensing). Download from the [Kaggle competition page](https://www.kaggle.com/c/home-credit-default-risk) and place CSVs in `data/raw/`.

## What's next
- Train/test split + baseline model
- Model comparison (Logistic Regression, Random Forest, XGBoost)
- Evaluation metrics & tuning
- FastAPI prediction endpoint
- Dockerize + deploy on AWS