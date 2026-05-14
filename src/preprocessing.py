# sklearn preprocessing pipeline

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import TARGET_COL, TRAIN_CSV


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


