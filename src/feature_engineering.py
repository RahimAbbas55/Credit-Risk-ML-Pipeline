import pandas as pd
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
    return df