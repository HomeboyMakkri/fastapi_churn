from collections.abc import AsyncIterator, Callable
from typing import cast

import httpx2
import pandas as pd
import pytest

from src import main
from src.dataset import ChurnDataset
from src.model import ModelConfigurationError
from src.model_store import ChurnModelArtifact


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client() -> AsyncIterator[httpx2.AsyncClient]:
    main.app.dependency_overrides.clear()
    main.app.state.churn_dataset = None
    main.app.state.churn_model = None
    transport = httpx2.ASGITransport(app=main.app)

    async with httpx2.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as test_client:
        yield test_client

    main.app.dependency_overrides.clear()


def prediction_payload(valid_record: dict[str, object]) -> dict[str, object]:
    return {
        name: value
        for name, value in valid_record.items()
        if name != "churn"
    }


def assert_common_error_shape(payload: dict[str, object]) -> None:
    assert set(payload) == {"code", "message", "details"}
    assert isinstance(payload["code"], str)
    assert isinstance(payload["message"], str)
    assert "Traceback" not in str(payload)


@pytest.mark.parametrize(
    "change_payload",
    [
        lambda payload: payload.pop("monthly_fee"),
        lambda payload: payload.update({"unknown_feature": 1}),
        lambda payload: payload.update({"monthly_fee": "49.99"}),
    ],
    ids=["missing-feature", "extra-feature", "wrong-value-type"],
)
async def test_predict_validation_errors_use_common_contract(
    client: httpx2.AsyncClient,
    valid_record: dict[str, object],
    change_payload: Callable[[dict[str, object]], object],
) -> None:
    main.app.dependency_overrides[main.get_churn_model] = lambda: cast(
        ChurnModelArtifact,
        object(),
    )
    payload = prediction_payload(valid_record)
    change_payload(payload)

    response = await client.post("/predict", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert_common_error_shape(body)
    assert body["code"] == "request_validation_error"
    assert body["message"] == "Request data is invalid"
    assert isinstance(body["details"], list)


async def test_predict_rejects_empty_batch_with_common_contract(
    client: httpx2.AsyncClient,
) -> None:
    main.app.dependency_overrides[main.get_churn_model] = lambda: cast(
        ChurnModelArtifact,
        object(),
    )

    response = await client.post("/predict", json=[])

    assert response.status_code == 422
    body = response.json()
    assert_common_error_shape(body)
    assert body["code"] == "request_validation_error"


async def test_predict_without_model_returns_named_503_error(
    client: httpx2.AsyncClient,
    valid_record: dict[str, object],
) -> None:
    response = await client.post(
        "/predict",
        json=prediction_payload(valid_record),
    )

    assert response.status_code == 503
    assert response.json() == {
        "code": "model_not_trained",
        "message": "Churn model is not trained",
        "details": None,
    }


async def test_prediction_failure_hides_internal_error(
    client: httpx2.AsyncClient,
    valid_record: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main.app.dependency_overrides[main.get_churn_model] = lambda: cast(
        ChurnModelArtifact,
        object(),
    )

    def fail_prediction(*args: object, **kwargs: object) -> None:
        raise ValueError("private scikit-learn implementation detail")

    monkeypatch.setattr(main, "predict_churn_batch", fail_prediction)

    response = await client.post(
        "/predict",
        json=prediction_payload(valid_record),
    )

    assert response.status_code == 500
    assert response.json() == {
        "code": "prediction_failed",
        "message": "Could not calculate churn prediction",
        "details": None,
    }
    assert "private scikit-learn" not in response.text


class DatasetStub:
    def __init__(self, dataframe: pd.DataFrame | None) -> None:
        self._dataframe = dataframe

    @property
    def dataframe(self) -> pd.DataFrame:
        if self._dataframe is None:
            raise RuntimeError("Dataset is not loaded")
        return self._dataframe.copy(deep=True)


async def test_training_rejects_empty_dataset_with_named_error(
    client: httpx2.AsyncClient,
) -> None:
    dataset = cast(ChurnDataset, DatasetStub(pd.DataFrame()))
    main.app.dependency_overrides[main.get_dataset] = lambda: dataset

    response = await client.post(
        "/model/train",
        json={"model_type": "logreg", "hyperparameters": {}},
    )

    assert response.status_code == 503
    assert response.json() == {
        "code": "dataset_empty",
        "message": "Churn dataset is empty",
        "details": None,
    }


async def test_training_translates_data_preparation_error(
    client: httpx2.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = cast(
        ChurnDataset,
        DatasetStub(pd.DataFrame({"placeholder": [1]})),
    )
    main.app.dependency_overrides[main.get_dataset] = lambda: dataset

    def fail_preparation(*args: object, **kwargs: object) -> None:
        raise ValueError("invalid training columns")

    monkeypatch.setattr(main, "prepare_and_split", fail_preparation)

    response = await client.post(
        "/model/train",
        json={"model_type": "logreg", "hyperparameters": {}},
    )

    assert response.status_code == 422
    assert response.json() == {
        "code": "data_preparation_error",
        "message": "Training dataset cannot be prepared",
        "details": {"reason": "invalid training columns"},
    }


async def test_training_translates_model_configuration_error(
    client: httpx2.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = cast(
        ChurnDataset,
        DatasetStub(pd.DataFrame({"placeholder": [1]})),
    )
    main.app.dependency_overrides[main.get_dataset] = lambda: dataset
    split = (
        pd.DataFrame({"feature": [1]}),
        pd.DataFrame({"feature": [1]}),
        pd.Series([0]),
        pd.Series([0]),
    )
    monkeypatch.setattr(main, "prepare_and_split", lambda dataframe: split)

    def fail_training(*args: object, **kwargs: object) -> None:
        raise ModelConfigurationError("unsupported hyperparameters")

    monkeypatch.setattr(main, "train_churn_model", fail_training)

    response = await client.post(
        "/model/train",
        json={"model_type": "logreg", "hyperparameters": {}},
    )

    assert response.status_code == 422
    assert response.json() == {
        "code": "model_configuration_error",
        "message": "Model configuration is invalid",
        "details": {"reason": "unsupported hyperparameters"},
    }


def test_train_and_predict_document_common_error_responses() -> None:
    openapi = main.app.openapi()

    for path in ("/predict", "/model/train"):
        operation = openapi["paths"][path]["post"]
        for status_code in ("422", "500", "503"):
            response = operation["responses"][status_code]
            schema = response["content"]["application/json"]["schema"]
            assert schema == {"$ref": "#/components/schemas/ErrorResponse"}
            assert "example" in response["content"]["application/json"]

    error_schema = openapi["components"]["schemas"]["ErrorResponse"]
    assert error_schema["additionalProperties"] is False
    assert set(error_schema["required"]) == {"code", "message"}
