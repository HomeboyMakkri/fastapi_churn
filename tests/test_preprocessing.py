from typing import Any

import pandas as pd
import pytest

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    prepare_and_split,
    prepare_features_and_target,
)


def make_dataframe(class_zero_rows: int = 8, class_one_rows: int = 4) -> pd.DataFrame:
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


def test_feature_groups_are_complete_unique_and_exclude_target() -> None:
    assert len(FEATURES) == 9
    assert len(set(FEATURES)) == len(FEATURES)
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
    assert TARGET_COLUMN not in FEATURES


def test_rejects_non_dataframe_input() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        prepare_features_and_target([])  # type: ignore[arg-type]


def test_prepares_features_and_target_without_mutating_input() -> None:
    dataframe = make_dataframe()
    original = dataframe.copy(deep=True)

    features, target = prepare_features_and_target(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)
    assert list(features.columns) == list(FEATURES)
    assert target.name == TARGET_COLUMN
    assert len(features) == len(target) == len(dataframe)


def test_removes_rows_with_missing_values_without_mutating_input() -> None:
    dataframe = make_dataframe()
    dataframe.loc[0, "region"] = None

    features, target = prepare_features_and_target(dataframe)

    assert len(features) == len(target) == len(dataframe) - 1
    assert not features.isna().any().any()
    assert pd.isna(dataframe.loc[0, "region"])


def test_keeps_duplicate_rows() -> None:
    dataframe = make_dataframe()
    duplicated = pd.concat([dataframe, dataframe.iloc[[0]]], ignore_index=True)

    features, target = prepare_features_and_target(duplicated)

    assert len(features) == len(target) == len(duplicated)


@pytest.mark.parametrize("column_problem", ["missing", "unexpected"])
def test_rejects_invalid_columns(column_problem: str) -> None:
    dataframe = make_dataframe()
    if column_problem == "missing":
        dataframe = dataframe.drop(columns=["region"])
    else:
        dataframe["customer_id"] = range(len(dataframe))

    with pytest.raises(ValueError, match=column_problem):
        prepare_features_and_target(dataframe)


def test_rejects_dataset_empty_after_missing_rows_are_removed() -> None:
    dataframe = make_dataframe()
    dataframe["monthly_fee"] = float("nan")

    with pytest.raises(ValueError, match="no rows after removing missing values"):
        prepare_features_and_target(dataframe)


@pytest.mark.parametrize(
    ("class_zero_rows", "class_one_rows", "message"),
    [
        (6, 0, "both churn classes"),
        (6, 1, "at least two rows"),
    ],
)
def test_rejects_target_that_cannot_be_stratified(
    class_zero_rows: int,
    class_one_rows: int,
    message: str,
) -> None:
    dataframe = make_dataframe(class_zero_rows, class_one_rows)

    with pytest.raises(ValueError, match=message):
        prepare_and_split(dataframe)


def test_rejects_invalid_target_value() -> None:
    dataframe = make_dataframe()
    dataframe.loc[0, TARGET_COLUMN] = 2

    with pytest.raises(ValueError, match="invalid values"):
        prepare_features_and_target(dataframe)


def test_creates_reproducible_stratified_split() -> None:
    dataframe = make_dataframe(class_zero_rows=16, class_one_rows=4)

    first_split = prepare_and_split(dataframe)
    second_split = prepare_and_split(dataframe)
    X_train, X_test, y_train, y_test = first_split

    assert X_train.shape == (16, 9)
    assert X_test.shape == (4, 9)
    assert y_train.value_counts().to_dict() == {0: 13, 1: 3}
    assert y_test.value_counts().to_dict() == {0: 3, 1: 1}
    for first_part, second_part in zip(first_split, second_split, strict=True):
        assert first_part.index.tolist() == second_part.index.tolist()


@pytest.mark.parametrize("test_size", [0, 1, -0.1, 1.1])
def test_rejects_test_size_outside_open_unit_interval(test_size: float) -> None:
    with pytest.raises(ValueError, match="greater than 0 and less than 1"):
        prepare_and_split(make_dataframe(), test_size=test_size)


def test_rejects_split_too_small_for_all_classes() -> None:
    with pytest.raises(ValueError, match="too few rows"):
        prepare_and_split(make_dataframe(2, 2), test_size=0.2)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("test_size", True, "test_size must be a number"),
        ("random_state", 4.2, "random_state must be an integer"),
    ],
)
def test_rejects_invalid_split_argument_types(
    argument: str,
    value: object,
    message: str,
) -> None:
    arguments = {argument: value}

    with pytest.raises(TypeError, match=message):
        prepare_and_split(make_dataframe(), **arguments)  # type: ignore[arg-type]
