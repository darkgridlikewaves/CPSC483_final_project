from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from src.evaluate import evaluate_regression
from src.preprocessing import (
    build_preprocessing_pipeline,
    get_feature_lists,
    load_and_split,
)
from src.train_regression import train_regression_models


def main():
    for d in (MODELS_DIR, FIGURES_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading and splitting data...")
    X_train, X_test, y_train, y_test = load_and_split()

    num_cols, cat_cols = get_feature_lists(X_train)
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)

    print("[2/3] Training regression models...")
    reg_models = train_regression_models(preprocessor, X_train, y_train)

    print("[3/3] Evaluating...")
    reg_metrics = evaluate_regression(reg_models, X_test, y_test)

    print("\nRegression metrics:")
    print(reg_metrics.to_string(index=False))
    print(f"\nModels saved to: {MODELS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Reports saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
