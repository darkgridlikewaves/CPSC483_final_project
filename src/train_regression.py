import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR, RANDOM_SEED


def train_regression_models(preprocessor, X_train, y_train):
    """
    Fit Ridge and Random Forest regressors with default hyperparameters.
    Returns a dict mapping model name -> fitted Pipeline.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "ridge": Pipeline([
            ("preprocessor", preprocessor),
            ("model", Ridge(random_state=RANDOM_SEED)),
        ]),
        "random_forest_reg": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
    }

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, MODELS_DIR / f"{name}.joblib")

    return models
