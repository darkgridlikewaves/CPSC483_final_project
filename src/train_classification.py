import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import CV_FOLDS, MODELS_DIR, RANDOM_SEED

PARAM_GRIDS = {
    "logistic_regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "model__max_iter": [1000],
    },
    "random_forest_clf": {
        "model__n_estimators": [100, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 4],
    },
}


def train_classification_models(preprocessor, X_train, y_train):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    base_models = {
        "logistic_regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(random_state=RANDOM_SEED)),
        ]),
        "random_forest_clf": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
    }

    best_models = {}
    for name, pipe in base_models.items():
        search = GridSearchCV(
            pipe,
            PARAM_GRIDS[name],
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best_models[name] = search.best_estimator_
        joblib.dump(search.best_estimator_, MODELS_DIR / f"{name}.joblib")
        print(f"{name} best params: {search.best_params_}")

    return best_models