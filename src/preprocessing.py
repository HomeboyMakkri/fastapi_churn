"""Prepare validated churn data for model training and evaluation."""

from math import ceil

import pandas as pd
from sklearn.model_selection import train_test_split

from .dataset_contract import CHURN_DATASET_CONTRACT


TARGET_COLUMN = CHURN_DATASET_CONTRACT.target
NUMERIC_FEATURES = CHURN_DATASET_CONTRACT.numeric_features
CATEGORICAL_FEATURES = CHURN_DATASET_CONTRACT.categorical_features
FEATURES = CHURN_DATASET_CONTRACT.features

FeatureTarget = tuple[pd.DataFrame, pd.Series]
DatasetSplit = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


def prepare_features_and_target(dataframe: pd.DataFrame) -> FeatureTarget:
    """Clean a churn dataframe and separate its features from the target.

    Rows containing missing values are removed. The input dataframe is never
    modified, and duplicate rows are retained because they can represent
    distinct customers when the dataset has no customer identifier.
    """
    CHURN_DATASET_CONTRACT.validate_columns(dataframe)

    cleaned = dataframe.dropna(
        subset=list(CHURN_DATASET_CONTRACT.columns)
    ).copy(deep=True)
    if cleaned.empty:
        raise ValueError("Dataset contains no rows after removing missing values")

    target = cleaned.loc[:, TARGET_COLUMN].copy(deep=True)
    target_values = set(target.unique())
    invalid_values = target_values - {0, 1}
    if invalid_values:
        formatted_values = sorted((repr(value) for value in invalid_values))
        raise ValueError(f"Target column contains invalid values: {formatted_values}")

    if len(target_values) < 2:
        raise ValueError("Target column must contain both churn classes: 0 and 1")

    features = cleaned.loc[:, list(FEATURES)].copy(deep=True)
    return features, target


def prepare_and_split(
    dataframe: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """Prepare churn data and create a reproducible stratified train/test split."""
    if isinstance(test_size, bool) or not isinstance(test_size, (int, float)):
        raise TypeError("test_size must be a number between 0 and 1")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise TypeError("random_state must be an integer")

    features, target = prepare_features_and_target(dataframe)
    class_counts = target.value_counts()
    if int(class_counts.min()) < 2:
        raise ValueError("Each churn class must contain at least two rows")

    test_rows = ceil(len(target) * test_size)
    train_rows = len(target) - test_rows
    class_count = len(class_counts)
    if test_rows < class_count or train_rows < class_count:
        raise ValueError(
            "test_size leaves too few rows to represent every class "
            "in both train and test sets"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    return X_train, X_test, y_train, y_test


def get_class_distribution(target: pd.Series) -> dict[str, int]:
    """Return counts for both churn classes using JSON-compatible keys."""
    _validate_distribution_target(target)
    counts = target.value_counts()
    return {
        str(churn_class): int(counts.get(churn_class, 0))
        for churn_class in (0, 1)
    }


def get_class_percentage(target: pd.Series) -> dict[str, float]:
    """Return percentages for both churn classes."""
    distribution = get_class_distribution(target)
    total = len(target)

    return {
        churn_class: round(count / total * 100, 2)
        for churn_class, count in distribution.items()
    }


def _validate_distribution_target(target: pd.Series) -> None:
    if not isinstance(target, pd.Series):
        raise TypeError("target must be a pandas Series")
    if target.empty:
        raise ValueError("Cannot calculate class distribution for an empty target")
    if target.isna().any():
        raise ValueError("Target contains missing values")

    invalid_values = set(target.unique()) - {0, 1}
    if invalid_values:
        formatted_values = sorted((repr(value) for value in invalid_values))
        raise ValueError(f"Target contains invalid values: {formatted_values}")
