import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import Request

from src import main
from src.schemas import TrainingHistoryEntry, TrainingMetrics
from src.training_history import (
    TrainingHistoryPersistenceError,
    save_training_history,
)


def make_entry(
    hour: int,
    model_type: str,
) -> TrainingHistoryEntry:
    return TrainingHistoryEntry(
        trained_at=datetime(2026, 9, 6, hour, 0, tzinfo=timezone.utc),
        model_type=model_type,  # type: ignore[arg-type]
        hyperparameters={"run": hour},
        metrics=TrainingMetrics(
            accuracy=0.70 + hour / 100,
            f1=0.40 + hour / 100,
            roc_auc=0.60 + hour / 100,
        ),
    )


@pytest.fixture
def history_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    path = tmp_path / "training_history.json"
    monkeypatch.setattr(main, "TRAINING_HISTORY_PATH", path)
    return path


def test_model_metrics_returns_empty_response_without_history(
    history_path: Path,
) -> None:
    response = main.get_model_metrics(limit=10, model_type=None)

    assert not history_path.exists()
    assert response.model_dump() == {"latest": None, "history": []}


def test_model_metrics_returns_latest_entries_first_and_applies_limit(
    history_path: Path,
) -> None:
    entries = [
        make_entry(1, "logreg"),
        make_entry(2, "random_forest"),
        make_entry(3, "logreg"),
    ]
    save_training_history(entries, history_path)

    response = main.get_model_metrics(limit=2, model_type=None)

    assert response.latest == entries[2]
    assert response.history == [entries[2], entries[1]]


@pytest.mark.parametrize(
    ("model_type", "expected_indexes"),
    [
        ("logreg", [2, 0]),
        ("random_forest", [1]),
    ],
)
def test_model_metrics_filters_history_by_model_type(
    history_path: Path,
    model_type: str,
    expected_indexes: list[int],
) -> None:
    entries = [
        make_entry(1, "logreg"),
        make_entry(2, "random_forest"),
        make_entry(3, "logreg"),
    ]
    save_training_history(entries, history_path)

    response = main.get_model_metrics(
        limit=10,
        model_type=model_type,  # type: ignore[arg-type]
    )

    expected = [entries[index] for index in expected_indexes]
    assert response.latest == expected[0]
    assert response.history == expected


def test_model_metrics_returns_empty_response_for_filter_without_matches(
    history_path: Path,
) -> None:
    save_training_history([make_entry(1, "logreg")], history_path)

    response = main.get_model_metrics(limit=10, model_type="random_forest")

    assert response.model_dump() == {"latest": None, "history": []}


def test_model_metrics_propagates_invalid_history_error(
    history_path: Path,
) -> None:
    history_path.write_text("not valid JSON", encoding="utf-8")

    with pytest.raises(TrainingHistoryPersistenceError, match="Could not load"):
        main.get_model_metrics(limit=10, model_type=None)


@pytest.mark.anyio
async def test_invalid_history_is_hidden_by_global_error_handler(
    history_path: Path,
) -> None:
    history_path.write_text("not valid JSON", encoding="utf-8")
    request = cast(
        Request,
        SimpleNamespace(
            method="GET",
            url=SimpleNamespace(path="/model/metrics"),
        ),
    )

    with pytest.raises(TrainingHistoryPersistenceError) as raised:
        main.get_model_metrics(limit=10, model_type=None)
    response = await main.unhandled_exception_handler(request, raised.value)

    assert response.status_code == 500
    assert json.loads(bytes(response.body)) == {
        "code": "internal_server_error",
        "message": "An unexpected server error occurred",
        "details": None,
    }


def test_model_metrics_is_documented_in_openapi() -> None:
    openapi = main.app.openapi()
    operation = openapi["paths"]["/model/metrics"]["get"]
    parameters = {
        parameter["name"]: parameter for parameter in operation["parameters"]
    }

    assert operation["summary"] == "Get churn model training metrics"
    assert operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/ModelMetricsResponse"}
    assert operation["responses"]["500"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/ErrorResponse"}
    assert parameters["limit"]["schema"] == {
        "type": "integer",
        "maximum": 100,
        "minimum": 1,
        "default": 10,
        "description": "Maximum number of training records to return",
        "title": "Limit",
    }
    assert parameters["model_type"]["schema"]["anyOf"] == [
        {"type": "string", "enum": ["logreg", "random_forest"]},
        {"type": "null"},
    ]
