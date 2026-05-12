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

# Price tier boundaries (based on dataset quartiles)
# Tier 0: Low (<130k), 1: Medium (130k-163k), 2: High (163k-215k), 3: Luxury (>215k)
TIER_BOUNDARIES = [130_000, 163_000, 215_000]
TIER_LABELS = ["Low", "Medium", "High", "Luxury"]
