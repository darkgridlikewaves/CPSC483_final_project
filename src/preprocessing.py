# sklearn preprocessing pipeline

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    RANDOM_SEED,
    TARGET_COL,
    TIER_BOUNDARIES,
    TRAIN_CSV,
    USE_LOG_TARGET,
)


def load_data(csv_path=TRAIN_CSV):
    """Load raw CSV and drop the Id column."""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["Id"])
    return df


def get_feature_lists(df):
    """
    Split dataframe columns into numerical and categorical lists.
    Excludes the target column automatically.

    Returns:
        numerical_cols (list): columns with numeric dtype
        categorical_cols (list): columns with object dtype
    """
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    numerical_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(df[c])]
    return numerical_cols, categorical_cols


def build_preprocessing_pipeline(numerical_cols, categorical_cols):
    """
    Build a ColumnTransformer that handles both numerical and categorical features.

    Returns:
        sklearn ColumnTransformer (unfitted)
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])

    return preprocessor


def make_tier_labels(y_raw):
    """
    Convert raw SalePrice values into integer tier labels (0-3).
    """
    tiers = pd.cut(
        y_raw,
        bins=TIER_BOUNDARIES,
        labels=[0, 1, 2, 3],
    )
    return tiers.astype(int)


def load_and_split(csv_path=TRAIN_CSV, test_size=0.2):
    df = load_data(csv_path)

    y_raw = df[TARGET_COL]
    y_reg = np.log1p(y_raw) if USE_LOG_TARGET else y_raw.values
    y_clf = make_tier_labels(y_raw)

    X = df.drop(columns=[TARGET_COL])

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y_clf,
    )

    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test
