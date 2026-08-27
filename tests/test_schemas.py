from typing import Any

import pytest
from pydantic import ValidationError

from src.schemas import DatasetRowChurn, FeatureVectorChurn


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


@pytest.mark.parametrize("churn", [-1, 2])
def test_dataset_row_rejects_invalid_target(
    valid_record: dict[str, object],
    churn: int,
) -> None:
    valid_record["churn"] = churn

    with pytest.raises(ValidationError):
        DatasetRowChurn.model_validate(valid_record)
