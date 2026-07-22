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