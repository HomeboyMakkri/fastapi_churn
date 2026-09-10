from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import Request

from src.application import create_app
from src.routers.system import get_health
from src.schemas import HealthStatus


def make_request(
    *,
    model_available: bool,
    dataset_loaded: bool,
) -> Request:
    state = SimpleNamespace(
        churn_model=object() if model_available else None,
        churn_dataset=object() if dataset_loaded else None,
    )
    return cast(
        Request,
        SimpleNamespace(app=SimpleNamespace(state=state)),
    )


@pytest.mark.parametrize(
    ("model_available", "dataset_loaded"),
    [
        (True, True),
        (False, True),
        (True, False),
        (False, False),
    ],
    ids=["both", "dataset-only", "model-only", "neither"],
)
def test_health_reports_runtime_resource_availability(
    model_available: bool,
    dataset_loaded: bool,
) -> None:
    response = get_health(
        make_request(
            model_available=model_available,
            dataset_loaded=dataset_loaded,
        )
    )

    assert response == HealthStatus(
        model_available=model_available,
        dataset_loaded=dataset_loaded,
    )


def test_health_handles_uninitialized_app_state() -> None:
    request = cast(
        Request,
        SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace())),
    )

    response = get_health(request)

    assert response == HealthStatus(
        model_available=False,
        dataset_loaded=False,
    )


def test_health_is_documented_in_openapi() -> None:
    openapi = create_app().openapi()
    operation = openapi["paths"]["/health"]["get"]
    response_schema = operation["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    health_schema = openapi["components"]["schemas"]["HealthStatus"]

    assert operation["summary"] == "Get churn service health"
    assert operation["description"]
    assert response_schema == {"$ref": "#/components/schemas/HealthStatus"}
    assert health_schema["additionalProperties"] is False
    assert set(health_schema["required"]) == {
        "model_available",
        "dataset_loaded",
    }
    assert {
        name: schema["type"]
        for name, schema in health_schema["properties"].items()
    } == {
        "model_available": "boolean",
        "dataset_loaded": "boolean",
    }
