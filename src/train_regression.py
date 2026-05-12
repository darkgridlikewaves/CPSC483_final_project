# Trains and tunes two regression models: Ridge and Random Forest using GridSearchCV.
# preprocessing is fitted only on training folds — no data leakage.
# Best models are saved to models/regression/.

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from src.config import CV_FOLDS, MODELS_DIR, RANDOM_SEED


def _get_cv():
    return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)


def train_ridge(preprocessor, X_train, y_train):
    """
    Train a Ridge Regression model using GridSearchCV.

    Searches over alpha values to find the best L2 regularization strength.
    Saves the best estimator to models/regression/ridge_best.pkl.

    Returns the fitted GridSearchCV object.
    """
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge()),
    ])

    param_grid = {
        "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    }

    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=_get_cv(),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)

    print(f"\n[Ridge] Best params : {gs.best_params_}")
    print(f"[Ridge] Best CV RMSE: {-gs.best_score_:.4f}")

    save_dir = MODELS_DIR / "regression"
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(gs.best_estimator_, save_dir / "ridge_best.pkl")
    print(f"[Ridge] Model saved to {save_dir / 'ridge_best.pkl'}")

    return gs


def train_random_forest_regressor(preprocessor, X_train, y_train):
    """
    Train a Random Forest Regressor using GridSearchCV.

    Searches over n_estimators, max_depth, and max_features.
    Saves the best estimator to models/regression/rf_regressor_best.pkl.

    Returns the fitted GridSearchCV object.
    """
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=RANDOM_SEED)),
    ])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__max_features": ["sqrt", "log2"],
    }

    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=_get_cv(),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)

    print(f"\n[RF Regressor] Best params : {gs.best_params_}")
    print(f"[RF Regressor] Best CV RMSE: {-gs.best_score_:.4f}")

    save_dir = MODELS_DIR / "regression"
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(gs.best_estimator_, save_dir / "rf_regressor_best.pkl")
    print(f"[RF Regressor] Model saved to {save_dir / 'rf_regressor_best.pkl'}")

    return gs


def run_regression_training(preprocessor, X_train, y_train):
    """
    Train both regression models and return their GridSearchCV objects.

    Returns:
        dict with keys 'ridge' and 'rf_regressor'
    """
    print("=" * 60)
    print("REGRESSION TRAINING: Ridge vs Random Forest")
    print("=" * 60)

    gs_ridge = train_ridge(preprocessor, X_train, y_train)
    gs_rf = train_random_forest_regressor(preprocessor, X_train, y_train)

    return {"ridge": gs_ridge, "rf_regressor": gs_rf}
