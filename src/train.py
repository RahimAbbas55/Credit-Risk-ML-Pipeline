"""
train.py

Trains and evaluates the credit risk prediction model.
XGBoost selected as final model after comparison against Logistic Regression
and Random Forest (see notebooks/03_model_comparison.ipynb for full analysis).

Run directly: python -m src.train
"""

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

from src.data_ingestion import load_application_train, load_bureau, load_previous_application
from src.feature_engineering import engineer_features_pipeline, prepare_model_data


def train_final_model():
    print("Loading data...")
    df_raw = load_application_train()
    bureau_raw = load_bureau()
    prev_app_raw = load_previous_application()

    print("Engineering features...")
    df = engineer_features_pipeline(df_raw, bureau_raw, prev_app_raw)

    print("Preparing model data...")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_model_data(df)

    print("Training XGBoost (tuned, final model)...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n--- Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, 'models/xgboost_tuned_final.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    print("\nModel and scaler saved to models/")

    return model, scaler


if __name__ == "__main__":
    train_final_model()