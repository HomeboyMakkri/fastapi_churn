from typing import Any, cast

import pandas as pd
import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.model import train_churn_model
from src.preprocessing import prepare_and_split


def make_dataframe(
    class_zero_rows: int = 20,
    class_one_rows: int = 10,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for index in range(class_zero_rows + class_one_rows):
        records.append(
            {
                "monthly_fee": 9.99 + index,
                "usage_hours": 10.0 + index,
                "support_requests": index % 3,
                "account_age_months": index + 1,
                "failed_payments": index % 2,
                "autopay_enabled": index % 2,
                "region": "europe" if index % 2 else "asia",
                "device_type": "mobile" if index % 2 else "desktop",
                "payment_method": "card" if index % 2 else "paypal",
                "churn": 0 if index < class_zero_rows else 1,
            }
        )

    return pd.DataFrame(records)


def make_learnable_dataframe(row_count: int = 200) -> pd.DataFrame:
    """Create balanced data where low usage is a strong churn signal."""
    records: list[dict[str, Any]] = []

    for index in range(row_count):
        churn = index % 2
        records.append(
            {
                "monthly_fee": 30.0,
                "usage_hours": 5.0 + index % 3 if churn else 45.0 + index % 3,
                "support_requests": 1,
                "account_age_months": 12,
                "failed_payments": 0,
                "autopay_enabled": 1,
                "region": "europe",
                "device_type": "mobile",
                "payment_method": "card",
                "churn": churn,
            }
        )

    return pd.DataFrame(records)


def test_train_churn_model_returns_fitted_expected_pipeline() -> None:
    X_train, _, y_train, _ = prepare_and_split(make_dataframe())

    pipeline = train_churn_model(X_train, y_train)

    assert isinstance(pipeline, Pipeline)
    check_is_fitted(pipeline)
    assert list(pipeline.named_steps) == ["preprocessing", "classifier"]

    preprocessor = pipeline.named_steps["preprocessing"]
    assert isinstance(preprocessor, ColumnTransformer)
    assert isinstance(preprocessor.named_transformers_["num"], StandardScaler)
    encoder = preprocessor.named_transformers_["cat"]
    assert isinstance(encoder, OneHotEncoder)
    
    encoder = preprocessor.named_transformers_['cat']
    assert encoder.get_params()['handle_unknown'] == "ignore"

    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)


def test_trained_model_predicts_classes_and_probabilities() -> None:
    X_train, X_test, y_train, _ = prepare_and_split(make_dataframe())
    pipeline = train_churn_model(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)

    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({0, 1})
    assert probabilities.shape == (len(X_test), 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert probabilities.sum(axis=1) == pytest.approx(1.0)


def test_pipeline_accepts_categories_unseen_during_training() -> None:
    X_train, X_test, y_train, _ = prepare_and_split(make_dataframe())
    pipeline = train_churn_model(X_train, y_train)
    unseen_customer = X_test.iloc[[0]].copy(deep=True)
    unseen_customer.loc[:, "region"] = "africa"
    unseen_customer.loc[:, "device_type"] = "tablet"
    unseen_customer.loc[:, "payment_method"] = "crypto"

    prediction = pipeline.predict(unseen_customer)

    prediction = cast(np.ndarray, pipeline.predict(unseen_customer))
    assert prediction.shape == (1,)
    
    assert prediction[0] in {0, 1}


def test_training_does_not_modify_input_data() -> None:
    X_train, _, y_train, _ = prepare_and_split(make_dataframe())
    original_features = X_train.copy(deep=True)
    original_target = y_train.copy(deep=True)

    train_churn_model(X_train, y_train)

    pd.testing.assert_frame_equal(X_train, original_features)
    pd.testing.assert_series_equal(y_train, original_target)


@pytest.mark.parametrize(
    ("features", "target", "message"),
    [
        (pd.DataFrame(), pd.Series(dtype="int64"), "must not be empty"),
        (
            pd.DataFrame({"feature": [1, 2]}),
            pd.Series([0]),
            "same number of rows",
        ),
        (
            pd.DataFrame({"feature": [1, 2]}),
            pd.Series([0, 0]),
            "both churn classes",
        ),
        (
            pd.DataFrame({"feature": [1, 2]}),
            pd.Series([0, 2]),
            "invalid values",
        ),
        (
            pd.DataFrame({"feature": [1, None]}),
            pd.Series([0, 1]),
            "missing values",
        ),
    ],
)
def test_train_churn_model_rejects_invalid_training_data(
    features: pd.DataFrame,
    target: pd.Series,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        train_churn_model(features, target)


@pytest.mark.parametrize(
    ("features", "target", "message"),
    [
        ([], pd.Series([0, 1]), "X_train must be a pandas DataFrame"),
        (
            pd.DataFrame({"feature": [1, 2]}),
            [0, 1],
            "y_train must be a pandas Series",
        ),
    ],
)
def test_train_churn_model_rejects_invalid_input_types(
    features: object,
    target: object,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        train_churn_model(features, target)  # type: ignore[arg-type]


def test_model_learns_a_clear_signal() -> None:
    X_train, X_test, y_train, y_test = prepare_and_split(
        make_learnable_dataframe()
    )
    pipeline = train_churn_model(X_train, y_train)

    predictions = pipeline.predict(X_test)

    assert accuracy_score(y_test, predictions) >= 0.95
    assert f1_score(y_test, predictions) >= 0.95
