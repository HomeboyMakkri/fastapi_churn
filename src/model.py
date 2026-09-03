"""Build and train configurable churn classification pipelines."""

from collections.abc import Mapping

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES


ChurnClassifier = LogisticRegression | RandomForestClassifier


class ModelConfigurationError(ValueError):
    """Raised when a churn classifier configuration is not supported."""


def create_churn_classifier(
    model_type: str,
    hyperparameters: Mapping[str, object] | None = None,
) -> ChurnClassifier:
    """Create a supported classifier with service defaults and overrides."""
    if model_type == "logreg":
        classifier: ChurnClassifier = LogisticRegression(max_iter=1000)
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(random_state=42)
    else:
        raise ModelConfigurationError(
            f"Unsupported model_type: {model_type!r}. "
            "Expected 'logreg' or 'random_forest'"
        )

    if hyperparameters is None:
        parameters: dict[str, object] = {}
    elif isinstance(hyperparameters, Mapping):
        parameters = dict(hyperparameters)
    else:
        raise ModelConfigurationError("hyperparameters must be a mapping")

    supported_parameters = classifier.get_params(deep=False)
    unsupported_parameters = sorted(set(parameters) - set(supported_parameters))
    if unsupported_parameters:
        formatted_parameters = ", ".join(unsupported_parameters)
        raise ModelConfigurationError(
            f"Unsupported hyperparameters for {model_type}: {formatted_parameters}"
        )

    try:
        classifier.set_params(**parameters)
    except (TypeError, ValueError) as error:
        raise ModelConfigurationError(
            f"Invalid hyperparameters for {model_type}: {error}"
        ) from error

    return classifier


def train_churn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "logreg",
    hyperparameters: Mapping[str, object] | None = None,
) -> Pipeline:
    """Fit and return a preprocessing and selected-classifier pipeline."""
    _validate_training_data(X_train, y_train)
    classifier = create_churn_classifier(model_type, hyperparameters)

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
            ("classifier", classifier),
        ]
    )

    try:
        pipeline.fit(X_train, y_train)
    except (TypeError, ValueError) as error:
        if hyperparameters:
            raise ModelConfigurationError(
                f"Invalid hyperparameters for {model_type}: {error}"
            ) from error
        raise

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
