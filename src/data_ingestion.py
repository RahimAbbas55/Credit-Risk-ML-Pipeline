import pandas as pd
from config import APPLICATION_TRAIN_PATH
from config import RAW_DATA_DIR

def load_application_train() -> pd.DataFrame:
    df = pd.read_csv(APPLICATION_TRAIN_PATH)
    print(f"Loaded application_train: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# Function to load the bureau.csv data : each applicant's credit history at different institutions
def load_bureau() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_DIR / "bureau.csv")
    print(f"Loaded bureau: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# Function to load the previous_application.csv data : each applicant's previous application history
def load_previous_application() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_DIR / "previous_application.csv")
    print(f"Loaded previous_application: {df.shape[0]} rows, {df.shape[1]} columns")
    return df