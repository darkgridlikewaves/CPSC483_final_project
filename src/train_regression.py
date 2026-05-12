import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from src.config import CV_FOLDS, MODELS_DIR, RANDOM_SEED

PARAM_GRIDS = {
    "ridge": {
        "model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0],
    },
    "random_forest_reg": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 10],
        "model__min_samples_leaf": [1, 4],
    },
}


def train_regression_models(preprocessor, X_train, y_train):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    base_models = {
        "ridge": Pipeline([
            ("preprocessor", preprocessor),
            ("model", Ridge()),
        ]),
        "random_forest_reg": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
    }

    best_models = {}
    for name, pipe in base_models.items():
        search = GridSearchCV(
            pipe,
            PARAM_GRIDS[name],
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best_models[name] = search.best_estimator_
        joblib.dump(search.best_estimator_, MODELS_DIR / f"{name}.joblib")
        print(f"{name} best params: {search.best_params_}")

    return best_models
