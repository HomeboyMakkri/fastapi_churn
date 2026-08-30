"""Shared structural contract for churn datasets."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetContract:
    """Describe feature roles and validate a dataframe's column structure."""

    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    target: str

    def __post_init__(self) -> None:
        columns = self.columns
        if not self.features:
            raise ValueError("Dataset contract must contain at least one feature")
        if not self.target:
            raise ValueError("Dataset contract target must not be empty")
        if any(not column for column in columns):
            raise ValueError("Dataset contract column names must not be empty")
        if len(set(columns)) != len(columns):
            raise ValueError("Dataset contract columns must be unique")

    @property
    def features(self) -> tuple[str, ...]:
        return self.numeric_features + self.categorical_features

    @property
    def columns(self) -> tuple[str, ...]:
        return (*self.features, self.target)

    def validate_columns(self, dataframe: pd.DataFrame) -> None:
        """Reject wrong, missing, unexpected, or duplicated columns."""
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")

        if dataframe.columns.has_duplicates:
            raise ValueError("Dataset contains duplicate column names")

        expected_columns = set(self.columns)
        actual_columns = set(dataframe.columns)
        missing_columns = expected_columns - actual_columns
        unexpected_columns = actual_columns - expected_columns

        if missing_columns or unexpected_columns:
            details: list[str] = []
            if missing_columns:
                details.append(f"missing: {sorted(missing_columns)}")
            if unexpected_columns:
                details.append(f"unexpected: {sorted(unexpected_columns)}")
            raise ValueError(f"Invalid dataset columns ({'; '.join(details)})")


CHURN_DATASET_CONTRACT = DatasetContract(
    numeric_features=(
        "monthly_fee",
        "usage_hours",
        "support_requests",
        "account_age_months",
        "failed_payments",
        "autopay_enabled",
    ),
    categorical_features=(
        "region",
        "device_type",
        "payment_method",
    ),
    target="churn",
)
