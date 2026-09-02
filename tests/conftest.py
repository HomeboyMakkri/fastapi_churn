from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.model import train_churn_model
from src.model_store import ChurnModelArtifact
from src.preprocessing import prepare_and_split


CsvFactory = Callable[[list[dict[str, object]]], Path]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def valid_record() -> dict[str, object]:
    return {
        "monthly_fee": 49.99,
        "usage_hours": 35.5,
        "support_requests": 2,
        "account_age_months": 18,
        "failed_payments": 1,
        "region": "europe",
        "device_type": "mobile",
        "payment_method": "card",
        "autopay_enabled": 1,
        "churn": 0,
    }


@pytest.fixture(scope="session")
def trained_artifact() -> ChurnModelArtifact:
    dataset_path = (
        Path(__file__).resolve().parent.parent / "data" / "churn_dataset.csv"
    )
    dataframe = pd.read_csv(dataset_path)
    X_train, _, y_train, _ = prepare_and_split(dataframe)

    return ChurnModelArtifact(
        pipeline=train_churn_model(X_train, y_train),
        trained_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
        accuracy=0.0,
        f1=0.0,
    )


@pytest.fixture
def csv_factory(tmp_path: Path) -> CsvFactory:
    def create_csv(records: list[dict[str, object]]) -> Path:
        path = tmp_path / "dataset.csv"
        pd.DataFrame(records).to_csv(path, index=False)
        return path

    return create_csv
