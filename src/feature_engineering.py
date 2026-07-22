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