import pathlib

# Paths
ROOT_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "kaggle_data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

TRAIN_CSV = DATA_DIR / "train.csv"

RANDOM_SEED = 42

TARGET_COL = "SalePrice"

USE_LOG_TARGET = True
