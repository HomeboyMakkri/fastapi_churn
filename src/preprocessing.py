"""Prepare validated churn data for model training and evaluation."""

from math import ceil

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "churn"

NUMERIC_FEATURES: tuple[str, ...] = (
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "region",
    "device_type",
    "payment_method",
)

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXPECTED_COLUMNS = frozenset((*FEATURES, TARGET_COLUMN))

FeatureTarget = tuple[pd.DataFrame, pd.Series]
DatasetSplit = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


def prepare_features_and_target(dataframe: pd.DataFrame) -> FeatureTarget:
    """Clean a churn dataframe and separate its features from the target.

    Rows containing missing values are removed. The input dataframe is never
    modified, and duplicate rows are retained because they can represent
    distinct customers when the dataset has no customer identifier.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")

    actual_columns = set(dataframe.columns)
    missing_columns = EXPECTED_COLUMNS - actual_columns
    unexpected_columns = actual_columns - EXPECTED_COLUMNS

    if missing_columns or unexpected_columns:
        details: list[str] = []
        if missing_columns:
            details.append(f"missing: {sorted(missing_columns)}")
        if unexpected_columns:
            details.append(f"unexpected: {sorted(unexpected_columns)}")
        raise ValueError(f"Invalid preprocessing columns ({'; '.join(details)})")

    cleaned = dataframe.dropna(subset=list(EXPECTED_COLUMNS)).copy(deep=True)
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
