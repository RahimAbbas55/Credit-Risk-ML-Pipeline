import pandas as pd
from config import APPLICATION_TRAIN_PATH

def load_application_train() -> pd.DataFrame:
    df = pd.read_csv(APPLICATION_TRAIN_PATH)
    print(f"Loaded application_train: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
