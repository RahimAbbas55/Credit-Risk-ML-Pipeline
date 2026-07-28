import json
import joblib
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Load artifacts once at import time (not per-request - these are expensive to load)
model = joblib.load(MODELS_DIR / "xgboost_tuned_final.joblib")
scaler = joblib.load(MODELS_DIR / "scaler.joblib")

with open(MODELS_DIR / "feature_defaults.json") as f:
    FEATURE_DEFAULTS = json.load(f)

MODEL_COLUMNS = list(FEATURE_DEFAULTS.keys())


def build_feature_vector(request_data: dict) -> pd.DataFrame:
    """
    Merge caller-provided fields with defaults, then one-hot encode
    categorical fields to match the model's expected columns exactly.
    """
    # Start from defaults, override with anything the caller actually provided
    row = FEATURE_DEFAULTS.copy()

    # Map simple numeric/categorical fields directly
    direct_fields = [
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH',
        'DAYS_EMPLOYED', 'CNT_CHILDREN', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
    ]
    for field in direct_fields:
        if request_data.get(field) is not None:
            row[field] = request_data[field]

    # Handle categorical fields - these need to become one-hot columns
    categorical_mapping = {
        'CODE_GENDER': request_data.get('CODE_GENDER'),
        'NAME_EDUCATION_TYPE': request_data.get('NAME_EDUCATION_TYPE'),
        'NAME_INCOME_TYPE': request_data.get('NAME_INCOME_TYPE'),
        'NAME_FAMILY_STATUS': request_data.get('NAME_FAMILY_STATUS'),
        'FLAG_OWN_CAR': request_data.get('FLAG_OWN_CAR'),
        'FLAG_OWN_REALTY': request_data.get('FLAG_OWN_REALTY'),
    }

    # Reset all one-hot columns for these categories to 0 first,
    # then set the one matching the caller's actual value to 1
    for prefix, value in categorical_mapping.items():
        if value is None:
            continue
        matching_cols = [c for c in MODEL_COLUMNS if c.startswith(f"{prefix}_")]
        for col in matching_cols:
            row[col] = 0
        target_col = f"{prefix}_{value}"
        if target_col in row:
            row[target_col] = 1
        # If the value doesn't match any known column (e.g. "Y" for FLAG columns
        # where "N" was the dropped reference category), it's implicitly handled
        # by leaving all matching columns at 0

    # Build DataFrame with columns in the exact order the model expects
    df = pd.DataFrame([row])[MODEL_COLUMNS]
    return df


def predict_default_risk(request_data: dict) -> dict:
    """
    Run a full prediction: build feature vector, scale, predict probability,
    and classify into a risk category.
    """
    X = build_feature_vector(request_data)
    X_scaled = scaler.transform(X)

    probability = float(model.predict_proba(X_scaled)[0, 1])
    prediction = int(probability >= 0.5)

    if probability < 0.15:
        risk_category = "Low"
    elif probability < 0.40:
        risk_category = "Medium"
    else:
        risk_category = "High"

    return {
        "default_probability": round(probability, 4),
        "risk_category": risk_category,
        "prediction": prediction
    }