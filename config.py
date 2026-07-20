from pathlib import Path

# Root of the project (this file's parent folder)
PROJECT_ROOT = Path(__file__).resolve().parent

# Data directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Specific files we know we'll need
APPLICATION_TRAIN_PATH = RAW_DATA_DIR / "application_train.csv"
APPLICATION_TEST_PATH = RAW_DATA_DIR / "application_test.csv"