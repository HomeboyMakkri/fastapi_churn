from collections.abc import AsyncIterator

import httpx2
import pytest

from src.main import app, lifespan


pytestmark = pytest.mark.anyio


@pytest.fixture
async def client() -> AsyncIterator[httpx2.AsyncClient]:
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
    response = await client.get("/dataset/preview", params={"count": count}) # type: ignore

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
