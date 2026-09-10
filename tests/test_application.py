from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
import pytest

from src import lifespan as lifespan_module
from src.application import create_app
from src.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
    unhandled_exception_handler,
)
from src.main import app as default_app


def test_create_app_returns_independent_fully_configured_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("resource loading must wait for lifespan")

    monkeypatch.setattr(lifespan_module, "ChurnDataset", fail_if_called)
    monkeypatch.setattr(lifespan_module, "load_churn_model", fail_if_called)

    first_app = create_app()
    second_app = create_app()

    assert first_app is not second_app
    assert first_app is not default_app
    assert first_app.state is not second_app.state
    first_app.state.marker = object()
    assert not hasattr(second_app.state, "marker")

    assert first_app.openapi() == default_app.openapi()
    assert second_app.openapi() == default_app.openapi()
    assert len(first_app.openapi()["paths"]) == 10
    assert first_app.exception_handlers[RequestValidationError] is (
        request_validation_exception_handler
    )
    assert first_app.exception_handlers[HTTPException] is http_exception_handler
    assert first_app.exception_handlers[Exception] is unhandled_exception_handler
