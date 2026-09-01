from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import httpx2
import pandas as pd
import pytest

from src.dataset import ChurnDataset
from src.main import app, get_dataset, lifespan
from src.model_store import ChurnModelArtifact, load_churn_model


pytestmark = pytest.mark.anyio


async def test_lifespan_keeps_service_available_without_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.main.DATASET_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr("src.main.MODEL_PATH", tmp_path / "missing.joblib")

    async with lifespan(app):
        assert app.state.churn_dataset is None
        assert app.state.churn_model is None


@pytest.fixture
async def client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> AsyncIterator[httpx2.AsyncClient]:
    monkeypatch.setattr("src.main.MODEL_PATH", tmp_path / "churn_model.joblib")
    transport = httpx2.ASGITransport(app=app)

    async with lifespan(app):
        async with httpx2.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as test_client:
            yield test_client


async def test_root_reports_running_service(client: httpx2.AsyncClient) -> None:
    response = await client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "ml churn server is running"}


async def test_predict_echoes_valid_feature_vector(
    client: httpx2.AsyncClient,
    valid_record: dict[str, object],
) -> None:
    valid_record.pop("churn")

    response = await client.post("/predict", json=valid_record)

    assert response.status_code == 200
    assert response.json() == valid_record


@pytest.mark.parametrize(
    "payload_change",
    [
        ("remove", "monthly_fee", None),
        ("replace", "region", "unknown"),
        ("add", "unknown_feature", 1),
    ],
)
async def test_predict_rejects_invalid_payload(
    client: httpx2.AsyncClient,
    valid_record: dict[str, object],
    payload_change: tuple[str, str, object],
) -> None:
    valid_record.pop("churn")
    operation, field, value = payload_change

    if operation == "remove":
        valid_record.pop(field)
    else:
        valid_record[field] = value

    response = await client.post("/predict", json=valid_record)

    assert response.status_code == 422


async def test_preview_uses_default_count(client: httpx2.AsyncClient) -> None:
    response = await client.get("/dataset/preview")

    assert response.status_code == 200
    assert len(response.json()) == 5
    assert set(response.json()[0]) == {
        "monthly_fee",
        "usage_hours",
        "support_requests",
        "account_age_months",
        "failed_payments",
        "region",
        "device_type",
        "payment_method",
        "autopay_enabled",
        "churn",
    }


async def test_preview_respects_requested_count(client: httpx2.AsyncClient) -> None:
    response = await client.get("/dataset/preview", params={"count": 12})

    assert response.status_code == 200
    assert len(response.json()) == 12


@pytest.mark.parametrize("count", [0, 101, "not-an-integer"])
async def test_preview_rejects_invalid_count(
    client: httpx2.AsyncClient,
    count: object,
) -> None:
    response = await client.get(
        "/dataset/preview",
        params={"count": count},  # type: ignore[dict-item]
    )

    assert response.status_code == 422


async def test_dataset_info_matches_dataset(client: httpx2.AsyncClient) -> None:
    response = await client.get("/dataset/info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_rows"] == 2000
    assert payload["total_columns"] == 10
    assert len(payload["column_names"]) == payload["total_columns"]
    assert payload["churn_distribution"] == {"0": 1597, "1": 403}
    assert payload["churn_percentage"] == {"0": 79.85, "1": 20.15}
    assert sum(payload["churn_distribution"].values()) == payload["total_rows"]


async def test_dataset_split_info_matches_stratified_split(
    client: httpx2.AsyncClient,
) -> None:
    response = await client.get("/dataset/split-info")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "train_rows": 1600,
        "test_rows": 400,
        "feature_count": 9,
        "train_churn_distribution": {"0": 1278, "1": 322},
        "test_churn_distribution": {"0": 319, "1": 81},
        "train_churn_percentage": {"0": 79.88, "1": 20.12},
        "test_churn_percentage": {"0": 79.75, "1": 20.25},
    }


async def test_model_train_returns_test_metrics(
    client: httpx2.AsyncClient,
) -> None:
    response = await client.post("/model/train")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "accuracy": pytest.approx(0.7875),
        "f1": pytest.approx(0.0449438202247191),
    }


async def test_model_train_persists_model_and_lifespan_restores_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "models" / "churn_model.joblib"
    monkeypatch.setattr("src.main.MODEL_PATH", model_path)
    transport = httpx2.ASGITransport(app=app)

    async with lifespan(app):
        async with httpx2.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as test_client:
            response = await test_client.post("/model/train")

        in_memory_artifact = app.state.churn_model
        assert response.status_code == 200
        assert isinstance(in_memory_artifact, ChurnModelArtifact)
        assert in_memory_artifact.trained_at.utcoffset() is not None

    restored_from_disk = load_churn_model(model_path)
    async with lifespan(app):
        restored_from_lifespan = app.state.churn_model

        assert isinstance(restored_from_lifespan, ChurnModelArtifact)
        assert restored_from_lifespan.trained_at == restored_from_disk.trained_at
        assert restored_from_lifespan.accuracy == pytest.approx(
            response.json()["accuracy"]
        )
        assert restored_from_lifespan.f1 == pytest.approx(response.json()["f1"])


async def test_model_status_reports_untrained_model(
    client: httpx2.AsyncClient,
) -> None:
    response = await client.get("/model/status")

    assert response.status_code == 200
    assert response.json() == {
        "is_trained": False,
        "last_trained_at": None,
        "metrics": None,
    }


async def test_model_status_reports_latest_training(
    client: httpx2.AsyncClient,
) -> None:
    training_response = await client.post("/model/train")
    status_response = await client.get("/model/status")

    assert training_response.status_code == 200
    assert status_response.status_code == 200
    payload = status_response.json()
    assert payload["is_trained"] is True
    assert payload["last_trained_at"].endswith("Z")
    assert payload["metrics"] == training_response.json()


async def test_model_train_returns_503_when_dataset_is_unavailable(
    client: httpx2.AsyncClient,
) -> None:
    dataset = app.state.churn_dataset
    del app.state.churn_dataset

    try:
        response = await client.post("/model/train")
    finally:
        app.state.churn_dataset = dataset

    assert response.status_code == 503
    assert response.json() == {"detail": "Churn dataset is not available"}


class DatasetStub:
    def __init__(self, dataframe: pd.DataFrame | None) -> None:
        self._dataframe = dataframe

    @property
    def dataframe(self) -> pd.DataFrame:
        if self._dataframe is None:
            raise RuntimeError("Dataset is not loaded")
        return self._dataframe.copy(deep=True)


@pytest.mark.parametrize(
    ("dataframe", "detail"),
    [
        (None, "Churn dataset is not loaded"),
        (pd.DataFrame(), "Churn dataset is empty"),
    ],
)
async def test_model_train_returns_503_for_unusable_dataset(
    client: httpx2.AsyncClient,
    dataframe: pd.DataFrame | None,
    detail: str,
) -> None:
    stub = cast(ChurnDataset, DatasetStub(dataframe))
    app.dependency_overrides[get_dataset] = lambda: stub

    try:
        response = await client.post("/model/train")
    finally:
        app.dependency_overrides.pop(get_dataset, None)

    assert response.status_code == 503
    assert response.json() == {"detail": detail}
