'''
train.py
Purpose: 
    In this file, the model training and evaluation will be done.
    Run directly: python/python3 src/train.py
'''

# Adding necessary imports
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

from src.data_ingestion import load_application_train, load_bureau, load_previous_application
from src.feature_engineering import engineer_features_pipeline, encode_categorical_features

def train_baseline_model():
    # 1. Loading  the data
    df_raw = load_application_train()
    bureau_raw = load_bureau()
    prev_app_raw = load_previous_application()
    
    # 2. Engineering features
    df = engineer_features_pipeline(df_raw , bureau_raw , prev_app_raw)
    
    # 3. Separating features and target variable
    y = df['TARGET']
    X = df.drop(columns=['TARGET' , 'SK_ID_CURR'])
    
    # 4. Performing one-hot encoding on the categorical features
    X_encoded = encode_categorical_features(X)
    
    # 5. Splitting the data into training and testing sets
    X_train , X_test , y_train , y_test = train_test_split(
        X_encoded , y ,
        test_size = 0.2,
        random_state = 42,
        stratify = y
    )
    
    # 6. Scaling the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. Training the logistic regression model
    model = LogisticRegression(
        max_iter = 1000,
        random_state = 42,
        class_weight = 'balanced'
    )
    model.fit(X_train_scaled , y_train)
    
    # 8. Evaluating the model
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 9. Saving the artifacts (model and scaler)
    joblib.dump(model, 'models/logistic_regression_baseline.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    print("\nModel and scaler saved to models/")
    
    return scaler , model

if __name__ == "__main__":
    train_baseline_model()