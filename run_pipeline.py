import numpy as np
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, TARGET_COL, USE_LOG_TARGET
from src.evaluate import evaluate_classification, evaluate_regression
from src.preprocessing import build_preprocessing_pipeline, get_feature_lists, load_data
from src.train_classification import train_classification_models
from src.train_regression import train_regression_models
from src.utils import make_tier_labels, print_section


def main():
    print_section("Loading Data")
    df = load_data()

    y_raw = df[TARGET_COL]
    y_reg = np.log1p(y_raw) if USE_LOG_TARGET else y_raw.values
    y_clf = make_tier_labels(y_raw)
    X = df.drop(columns=[TARGET_COL])

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )

    print("Building preprocessor...")
    numerical_cols, categorical_cols = get_feature_lists(df)
    preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols)

    print_section("Regression")
    reg_models = train_regression_models(preprocessor, X_train, y_reg_train)
    reg_metrics = evaluate_regression(reg_models, X_test, y_reg_test)
    print(reg_metrics.to_string(index=False))

    print_section("Classification")
    clf_models = train_classification_models(preprocessor, X_train, y_clf_train)
    clf_metrics = evaluate_classification(clf_models, X_test, y_clf_test)
    print(clf_metrics.to_string(index=False))

    print("\nDone. Models saved to models/, outputs saved to outputs/")


if __name__ == "__main__":
    main()
