import pandas as pd
import pytest

from src.dataset_contract import CHURN_DATASET_CONTRACT, DatasetContract
from src.schemas import DatasetRowChurn


def test_churn_contract_matches_pydantic_dataset_schema() -> None:
    assert set(CHURN_DATASET_CONTRACT.columns) == set(
        DatasetRowChurn.model_fields
    )


def test_churn_contract_separates_feature_roles_and_target() -> None:
    assert len(CHURN_DATASET_CONTRACT.features) == 9
    assert set(CHURN_DATASET_CONTRACT.numeric_features).isdisjoint(
        CHURN_DATASET_CONTRACT.categorical_features
    )
    assert CHURN_DATASET_CONTRACT.target not in CHURN_DATASET_CONTRACT.features


def test_contract_rejects_duplicate_dataframe_columns() -> None:
    dataframe = pd.DataFrame([[1, 2]], columns=["churn", "churn"])

    with pytest.raises(ValueError, match="duplicate column names"):
        CHURN_DATASET_CONTRACT.validate_columns(dataframe)


def test_contract_rejects_duplicate_declared_columns() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        DatasetContract(
            numeric_features=("feature",),
            categorical_features=("feature",),
            target="churn",
        )
