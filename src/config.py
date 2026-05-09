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
CV_FOLDS = 5

TARGET_COL = "SalePrice"

USE_LOG_TARGET = True

# Price Tier Boundaries:
# Based on Ames Housing dataset quartiles (~25% per class)
# Q1 ~$130k, Q2 ~$163k, Q3 ~$215k
TIER_BOUNDARIES = [0, 130_000, 163_000, 215_000, float("inf")]
TIER_LABELS = ["Low Tier", "Medium Tier", "High Tier", "Luxury"]
