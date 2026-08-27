from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest


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


@pytest.fixture
def csv_factory(tmp_path: Path) -> CsvFactory:
    def create_csv(records: list[dict[str, object]]) -> Path:
        path = tmp_path / "dataset.csv"
        pd.DataFrame(records).to_csv(path, index=False)
        return path

    return create_csv
