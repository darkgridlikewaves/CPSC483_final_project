"""
predict.py — CLI tool for making predictions on a single house.
"""

import argparse
import sys

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, TIER_LABELS, TRAIN_CSV
from src.preprocessing import load_data


def build_input_row(args, training_df):
    feature_cols = [c for c in training_df.columns if c != "SalePrice"]
    row = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(training_df[col]):
            row[col] = training_df[col].median()
        else:
            row[col] = training_df[col].mode()[0]

    overrides = {
        "OverallQual": args.overall_quality,
        "GrLivArea": args.grade_living_area,
        "GarageCars": args.garage_cars,
        "TotalBsmtSF": args.total_basement,
        "YearBuilt": args.year_built,
        "Neighborhood": args.neighborhood,
        "BedroomAbvGr": args.bedrooms,
        "FullBath": args.full_bath,
    }
    for col, val in overrides.items():
        if val is not None:
            row[col] = val

    return pd.DataFrame([row])


def predict_regression(X_input, model_name, verbose):
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        print(f"[ERROR] Model file not found: {path}")
        print("Run 'python run_pipeline.py' first to train and save models.")
        sys.exit(1)

    model = joblib.load(path)
    log_prediction = model.predict(X_input)[0]
    price = np.expm1(log_prediction)
    print(f"\n[Regression — {model_name}]")
    print(f"Predicted Sale Price: ${price:,.0f}")
    if verbose:
        print(f"(log-scale prediction: {log_prediction:.4f})")


def predict_classification(X_input, model_name, verbose):
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        print(f"[ERROR] Model file not found: {path}")
        print("Run 'python run_pipeline.py' first to train and save models.")
        sys.exit(1)

    model = joblib.load(path)
    tier_idx = model.predict(X_input)[0]
    tier_label = TIER_LABELS[tier_idx]
    print(f"\n[Classification — {model_name}]")
    print(f"Predicted Price Tier: {tier_label} (tier {tier_idx})")
    if verbose:
        probs = model.predict_proba(X_input)[0]
        print("  Confidence scores:")
        for label, prob in zip(TIER_LABELS, probs):
            print(f"    {label:8s}: {prob:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Predict house sale price or price tier.")
    parser.add_argument(
        "--overall-quality",
        type=int,
        help="Overall quality (1-10)"
    )
    parser.add_argument(
        "--grade-living-area",
        type=int,
        help="Above-grade living area in sq ft"
    )
    parser.add_argument(
        "--garage-cars",
        type=int,
        help="Garage capacity"
    )
    parser.add_argument(
        "--total-basement",
        type=int,
        help="Total basement area in sq ft"
    )
    parser.add_argument(
        "--year-built",
        type=int,
        help="Year house was built"
    )
    parser.add_argument(
        "--neighborhood",
        type=str,
        help="Neighborhood name"
    )
    parser.add_argument(
        "--bedrooms",
        type=int,
        help="Bedrooms above basement level"
    )
    parser.add_argument(
        "--full-bath",
        type=int,
        help="Number of full bathrooms"
    )
    parser.add_argument(
        "--task",
        choices=["regression", "classification", "both"],
        default="both"
    )
    parser.add_argument(
        "--model",
        choices=["ridge", "rf", "logistic"],
        default="rf",
        help="ridge or rf for regression; logistic or rf for classification"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show extra information"
    )
    args = parser.parse_args()

    reg_model = "ridge" if args.model == "ridge" else "random_forest_reg"
    clf_model = "logistic_regression" if args.model == "logistic" else "random_forest_clf"

    print("Loading training data to build feature defaults...")
    training_df = load_data(TRAIN_CSV)
    X_input = build_input_row(args, training_df)

    if args.task in ("regression", "both"):
        predict_regression(X_input, reg_model, args.verbose)

    if args.task in ("classification", "both"):
        predict_classification(X_input, clf_model, args.verbose)


if __name__ == "__main__":
    main()
