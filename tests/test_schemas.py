from datetime import datetime, timezone
from itertools import product
from typing import Any

import pytest
from pydantic import ValidationError

from src.schemas import (
    DatasetRowChurn,
    DatasetSplitInfo,
    FeatureVectorChurn,
    ModelStatus,
    ModelTrainingInfo,
    PredictionResponseChurn,
    TrainingConfigChurn,
)


def test_feature_vector_accepts_valid_data(valid_record: dict[str, object]) -> None:
    feature_data = {key: value for key, value in valid_record.items() if key != "churn"}

    feature_vector = FeatureVectorChurn.model_validate(feature_data)

    assert feature_vector.monthly_fee == 49.99
    assert feature_vector.region == "europe"
    assert feature_vector.autopay_enabled == 1


def test_dataset_row_includes_target(valid_record: dict[str, object]) -> None:
    row = DatasetRowChurn.model_validate(valid_record)

    assert row.churn == 0
    assert set(row.model_dump()) == set(DatasetRowChurn.model_fields)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("monthly_fee", -0.01),
        ("usage_hours", float("nan")),
        ("support_requests", -1),
        ("account_age_months", -1),
        ("failed_payments", -1),
        ("region", "antarctica"),
        ("device_type", "smart_tv"),
        ("payment_method", "cash"),
        ("autopay_enabled", 2),
    ],
)
def test_feature_vector_rejects_invalid_values(
    valid_record: dict[str, object],
    field: str,
    invalid_value: Any,
) -> None:
    feature_data = {key: value for key, value in valid_record.items() if key != "churn"}
    feature_data[field] = invalid_value

    with pytest.raises(ValidationError):
        FeatureVectorChurn.model_validate(feature_data)


def test_feature_vector_rejects_unknown_fields(
    valid_record: dict[str, object],
) -> None:
    feature_data = {key: value for key, value in valid_record.items() if key != "churn"}
    feature_data["unknown_feature"] = "unexpected"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        FeatureVectorChurn.model_validate(feature_data)


def test_prediction_response_accepts_binary_probabilities() -> None:
    response = PredictionResponseChurn(
        predicted_class=1,
        class_probabilities={"0": 0.25, "1": 0.75},
    )

    assert response.model_dump() == {
        "predicted_class": 1,
        "class_probabilities": {"0": 0.25, "1": 0.75},
    }


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ({"0": 1.0}, "classes 0 and 1"),
        ({"0": 0.7, "1": 0.4}, "sum to 1"),
        ({"0": -0.1, "1": 1.1}, "greater than or equal to 0"),
        ({"0": float("nan"), "1": 1.0}, "finite number"),
    ],
)
def test_prediction_response_rejects_invalid_probabilities(
    probabilities: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        PredictionResponseChurn(
            predicted_class=1,
            class_probabilities=probabilities,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("churn", [-1, 2])
def test_dataset_row_rejects_invalid_target(
    valid_record: dict[str, object],
    churn: int,
) -> None:
    valid_record["churn"] = churn

    with pytest.raises(ValidationError):
        DatasetRowChurn.model_validate(valid_record)


def test_dataset_split_info_accepts_valid_summary() -> None:
    split_info = DatasetSplitInfo(
        train_rows=80,
        test_rows=20,
        feature_count=9,
        train_churn_distribution={"0": 64, "1": 16},
        test_churn_distribution={"0": 16, "1": 4},
        train_churn_percentage={"0": 80.0, "1": 20.0},
        test_churn_percentage={"0": 80.0, "1": 20.0},
    )

    assert split_info.train_rows == 80
    assert split_info.feature_count == 9


@pytest.mark.parametrize("field", ["train_rows", "test_rows", "feature_count"])
def test_dataset_split_info_rejects_nonpositive_sizes(field: str) -> None:
    payload = {
        "train_rows": 80,
        "test_rows": 20,
        "feature_count": 9,
        "train_churn_distribution": {"0": 64, "1": 16},
        "test_churn_distribution": {"0": 16, "1": 4},
        "train_churn_percentage": {"0": 80.0, "1": 20.0},
        "test_churn_percentage": {"0": 80.0, "1": 20.0},
    }
    payload[field] = 0

    with pytest.raises(ValidationError):
        DatasetSplitInfo.model_validate(payload)


@pytest.mark.parametrize("model_type", ["logreg", "random_forest"])
def test_training_config_accepts_supported_model_types(model_type: str) -> None:
    config = TrainingConfigChurn(model_type=model_type)  # type: ignore[arg-type]

    assert config.model_type == model_type
    assert config.hyperparameters == {}


def test_training_config_accepts_scalar_hyperparameters() -> None:
    hyperparameters = {
        "solver": "liblinear",
        "max_iter": 500,
        "tolerance": 0.001,
        "fit_intercept": True,
        "class_weight": None,
    }

    config = TrainingConfigChurn(
        model_type="logreg",
        hyperparameters=hyperparameters,
    )

    assert config.hyperparameters == hyperparameters


def test_training_config_rejects_unknown_model_type() -> None:
    with pytest.raises(ValidationError):
        TrainingConfigChurn(model_type="svm")  # type: ignore[arg-type]


def test_training_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrainingConfigChurn.model_validate(
            {
                "model_type": "logreg",
                "unexpected": "value",
            }
        )


@pytest.mark.parametrize("nested_value", [[100, 200], {"min_samples": 2}])
def test_training_config_rejects_nested_hyperparameters(
    nested_value: object,
) -> None:
    with pytest.raises(ValidationError):
        TrainingConfigChurn.model_validate(
            {
                "model_type": "random_forest",
                "hyperparameters": {"nested": nested_value},
            }
        )


def test_model_training_info_accepts_valid_metrics() -> None:
    metrics = ModelTrainingInfo(accuracy=0.8, f1=0.5)

    assert metrics.model_dump() == {"accuracy": 0.8, "f1": 0.5}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("accuracy", -0.01),
        ("accuracy", 1.01),
        ("f1", -0.01),
        ("f1", 1.01),
    ],
)
def test_model_training_info_rejects_metrics_outside_unit_interval(
    field: str,
    value: float,
) -> None:
    payload = {"accuracy": 0.8, "f1": 0.5}
    payload[field] = value

    with pytest.raises(ValidationError):
        ModelTrainingInfo.model_validate(payload)


def test_model_status_accepts_untrained_state() -> None:
    model_status = ModelStatus(
        is_trained=False,
        last_trained_at=None,
        metrics=None,
        model_type=None,
        hyperparameters=None,
    )

    assert model_status.model_dump() == {
        "is_trained": False,
        "last_trained_at": None,
        "metrics": None,
        "model_type": None,
        "hyperparameters": None,
    }


@pytest.mark.parametrize(
    ("model_type", "hyperparameters"),
    [
        ("logreg", {}),
        (
            "random_forest",
            {"n_estimators": 50, "max_depth": 4, "random_state": 7},
        ),
    ],
)
def test_model_status_accepts_trained_state(
    model_type: str,
    hyperparameters: dict[str, object],
) -> None:
    trained_at = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)

    model_status = ModelStatus.model_validate(
        {
            "is_trained": True,
            "last_trained_at": trained_at,
            "metrics": {"accuracy": 0.8, "f1": 0.5},
            "model_type": model_type,
            "hyperparameters": hyperparameters,
        }
    )

    assert model_status.last_trained_at == trained_at
    assert model_status.metrics == ModelTrainingInfo(accuracy=0.8, f1=0.5)
    assert model_status.model_type == model_type
    assert model_status.hyperparameters == hyperparameters


@pytest.mark.parametrize(
    (
        "is_trained",
        "has_training_time",
        "has_metrics",
        "has_model_type",
        "has_hyperparameters",
    ),
    [
        (is_trained, *metadata_presence)
        for is_trained in (False, True)
        for metadata_presence in product((False, True), repeat=4)
        if (is_trained, metadata_presence)
        not in {
            (False, (False, False, False, False)),
            (True, (True, True, True, True)),
        }
    ],
)
def test_model_status_rejects_inconsistent_state(
    is_trained: bool,
    has_training_time: bool,
    has_metrics: bool,
    has_model_type: bool,
    has_hyperparameters: bool,
) -> None:
    with pytest.raises(ValidationError, match="metadata must match"):
        ModelStatus(
            is_trained=is_trained,
            last_trained_at=(
                datetime(2026, 9, 2, tzinfo=timezone.utc)
                if has_training_time
                else None
            ),
            metrics=(
                ModelTrainingInfo(accuracy=0.8, f1=0.5)
                if has_metrics
                else None
            ),
            model_type="logreg" if has_model_type else None,
            hyperparameters={} if has_hyperparameters else None,
        )
