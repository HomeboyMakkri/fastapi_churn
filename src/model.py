"""Build and train the baseline churn classification pipeline."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def train_churn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit and return a preprocessing and logistic-regression pipeline."""
    _validate_training_data(X_train, y_train)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                #class_weight='balanced',
                #random_state=42,
                )),
        ]
    )

    pipeline.fit(X_train, y_train)
    return pipeline


def _validate_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    """Validate invariants required to fit a binary churn classifier."""
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series")
    if X_train.empty or y_train.empty:
        raise ValueError("Training data must not be empty")
    if len(X_train) != len(y_train):
        raise ValueError("Features and target must contain the same number of rows")
    if X_train.isna().any().any() or y_train.isna().any():
        raise ValueError("Training data must not contain missing values")

    target_values = set(y_train.unique())
    invalid_values = target_values - {0, 1}
    if invalid_values:
        formatted_values = sorted(repr(value) for value in invalid_values)
        raise ValueError(f"Target contains invalid values: {formatted_values}")
    if len(target_values) < 2:
        raise ValueError("Target must contain both churn classes: 0 and 1")
