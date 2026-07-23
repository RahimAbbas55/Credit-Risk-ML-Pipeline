import pandas as pd

# Function to fix the "365243" anomaly in the DAYS_EMPLOYED column
def fix_days_employed_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Fix the DAYS_EMPLOYED anomaly where 365243 is used as a placeholder
    (likely for retired/unemployed applicants).
    Creates a flag column before replacing the placeholder with NaN.
    '''
    df = df.copy()
    ANOMALY_VALUE = 365243
    df['DAYS_EMPLOYED_ANOM'] = df['DAYS_EMPLOYED'] == ANOMALY_VALUE
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(ANOMALY_VALUE , pd.NA)
    df['DAYS_EMPLOYED'] = pd.to_numeric(df['DAYS_EMPLOYED'], errors='coerce')
    return df

# Function to fix the income anomaly where the income is extremely high (e.g., 1170000000)
def cap_income_anomaly(df: pd.DataFrame , cap_percentile: float = 0.99) -> pd.DataFrame:
    '''
    Cap extreme AMT_INCOME_TOTAL values at a given percentile to prevent
    a small number of outliers from distorting scaling and model training.
    '''
    df = df.copy()
    cap_value = df['AMT_INCOME_TOTAL'].quantile(cap_percentile)
    df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'].clip(upper = cap_value)
    return df

# Function to handle the missing value in the dataset
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Handle missing values using different strategies based on column type
    and reason for missingness.
    '''
    df = df.copy()
    # OWN_CAR_AGE: missing means no car.(structural missingness)
    df['OWN_CAR_AGE'] = df['OWN_CAR_AGE'].fillna(0)
    
    # Numeric columns: fill with median
    numeric_cols = df.select_dtypes(include = ['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
           df[col] = df[col].fillna(df[col].median())
    
    # Categorical columns: fill with 'Unknown' instead of dropping
    categorical_cols = df.select_dtypes(include = ['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna('Unknown')
            
    return df

# Wrapper function to call all functions in sequence 
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    un the full feature engineering pipeline: fix anomalies,
    cap outliers, and handle missing values.
    '''
    df = fix_days_employed_anomaly(df)
    df = cap_income_anomaly(df)
    df = handle_missing_values(df)
    return df

# Function to aggregate bureau.csv data into 1 row per SK_ID_CURR
def aggregate_bureau_data(bureau: pd.DataFrame) -> pd.DataFrame:
    agg = bureau.groupby('SK_ID_CURR').agg(
        BUREAU_CREDIT_COUNT=('SK_ID_BUREAU', 'count'),
        BUREAU_DAYS_CREDIT_MEAN=('DAYS_CREDIT', 'mean'),
        BUREAU_CREDIT_SUM_MEAN=('AMT_CREDIT_SUM', 'mean'),
        BUREAU_CREDIT_SUM_MAX=('AMT_CREDIT_SUM', 'max'),
        BUREAU_CREDIT_SUM_DEBT_MEAN=('AMT_CREDIT_SUM_DEBT', 'mean'),
        BUREAU_CREDIT_DAY_OVERDUE_MAX=('CREDIT_DAY_OVERDUE', 'max'),
        BUREAU_CREDIT_MAX_OVERDUE_MEAN=('AMT_CREDIT_MAX_OVERDUE', 'mean'),
        BUREAU_CNT_CREDIT_PROLONG_SUM=('CNT_CREDIT_PROLONG', 'sum'),
    ).reset_index()
    return agg

# Function to aggregate previous_application.csv data into 1 row per SK_ID_CURR
def build_previous_application_features(previous_application: pd.DataFrame) -> pd.DataFrame:
    previous_application = previous_application.copy()
    previous_application['IS_APPROVED'] = (previous_application['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
    previous_application['IS_REFUSED'] = (previous_application['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    agg = previous_application.groupby('SK_ID_CURR').agg(
        PREV_APP_COUNT=('SK_ID_PREV', 'count'),
        PREV_APP_APPROVAL_RATE=('IS_APPROVED', 'mean'),
        PREV_APP_REFUSAL_RATE=('IS_REFUSED', 'mean'),
        PREV_APP_CREDIT_MEAN=('AMT_CREDIT', 'mean'),
        PREV_APP_ANNUITY_MEAN=('AMT_ANNUITY', 'mean'),
        PREV_APP_DAYS_DECISION_MEAN=('DAYS_DECISION', 'mean'),
    ).reset_index()
    return agg

# Function to left-join bureau and previous_application aggregated features onto the main application table using SK_ID_CURR.
def merge_all_features(
    application: pd.DataFrame,
    bureau_features: pd.DataFrame,
    previous_application_features: pd.DataFrame    
) -> pd.DataFrame:
    df = application.merge(bureau_features , on = 'SK_ID_CURR' , how = 'left')
    df = df.merge(previous_application_features , on = 'SK_ID_CURR' , how = 'left')
    return df

# Handling new missing values and final verifications
def fill_missing_history_features(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    history_cols = [col for col in df.columns if col.startswith('BUREAU_') or col.startswith('PREV_APP_')]
    df[history_cols] = df[history_cols].fillna(0)
    return df

# Final feature engineering pipeline
def engineer_features_pipeline(df: pd.DataFrame , bureau: pd.DataFrame , prev_app: pd.DataFrame) -> pd.DataFrame:
    '''
    Run the full feature engineering pipeline: fix anomalies, cap outliers, handle missing values, 
    and merge bureau and previous application features.
    '''
    df = fix_days_employed_anomaly(df)
    df = cap_income_anomaly(df)
    bureau_features = aggregate_bureau_data(bureau)
    prev_app_features = build_previous_application_features(prev_app)
    df = merge_all_features(df , bureau_features , prev_app_features)
    df = fill_missing_history_features(df)
    df = handle_missing_values(df) 
    return df  