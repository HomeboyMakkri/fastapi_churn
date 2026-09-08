import logging
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from sklearn.pipeline import Pipeline

from src import main
from src.dataset import ChurnDataset
from src.errors import ModelNotTrainedError, PredictionError
from src.model_store import ChurnModelArtifact
from src.schemas import (
    FeatureVectorChurn,
    PredictionResponseChurn,
    TrainingConfigChurn,
    TrainingMetrics,
)


def logger_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == main.logger.name
    ]


def logger_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in logger_records(caplog)]


def make_request(app: object) -> Request:
    return cast(Request, SimpleNamespace(app=app))


def make_feature_vector(valid_record: dict[str, object]) -> FeatureVectorChurn:
    return FeatureVectorChurn.model_validate(
        {
            name: value
            for name, value in valid_record.items()
            if name != "churn"
        }
    )


class LoadedDatasetStub:
    def __init__(self, path: Path, row_count: int = 3) -> None:
        self.path = path
        self._dataframe = pd.DataFrame(index=range(row_count))

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe.copy(deep=True)

    def load(self) -> pd.DataFrame:
        return self.dataframe


class FailingDatasetStub:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> pd.DataFrame:
        raise ValueError("simulated invalid dataset")


class TrainingDatasetStub:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self._dataframe = dataframe

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe.copy(deep=True)


@pytest.mark.anyio
async def test_lifespan_logs_successful_dataset_and_model_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dataset_path = tmp_path / "churn_dataset.csv"
    model_path = tmp_path / "churn_model.joblib"
    artifact = ChurnModelArtifact(
        pipeline=Pipeline(steps=[]),
        trained_at=datetime(2026, 9, 8, tzinfo=timezone.utc),
        accuracy=0.8,
        f1=0.6,
        model_type="logreg",
        hyperparameters={},
    )
    dataset = LoadedDatasetStub(dataset_path)
    app_stub = SimpleNamespace(state=SimpleNamespace())
    monkeypatch.setattr(main, "DATASET_PATH", dataset_path)
    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    monkeypatch.setattr(main, "ChurnDataset", lambda _path: dataset)
    monkeypatch.setattr(main, "load_churn_model", lambda _path: artifact)
    caplog.set_level(logging.INFO, logger=main.logger.name)

    async with main.lifespan(cast(FastAPI, app_stub)):
        assert app_stub.state.churn_dataset is dataset
        assert app_stub.state.churn_model is artifact

    messages = logger_messages(caplog)
    assert any("Loading churn dataset" in message for message in messages)
    assert any("Loaded churn dataset" in message for message in messages)
    assert any("3 rows" in message for message in messages)
    assert any("Restoring churn model" in message for message in messages)
    assert any("Restored logreg churn model" in message for message in messages)


@pytest.mark.anyio
async def test_lifespan_logs_dataset_failure_and_missing_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dataset_path = tmp_path / "invalid.csv"
    model_path = tmp_path / "missing.joblib"
    dataset = FailingDatasetStub(dataset_path)
    app_stub = SimpleNamespace(state=SimpleNamespace())

    def raise_missing_model(_path: Path) -> ChurnModelArtifact:
        raise FileNotFoundError("simulated missing model")

    monkeypatch.setattr(main, "DATASET_PATH", dataset_path)
    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    monkeypatch.setattr(main, "ChurnDataset", lambda _path: dataset)
    monkeypatch.setattr(main, "load_churn_model", raise_missing_model)
    caplog.set_level(logging.INFO, logger=main.logger.name)

    async with main.lifespan(cast(FastAPI, app_stub)):
        assert app_stub.state.churn_dataset is None
        assert app_stub.state.churn_model is None

    records = logger_records(caplog)
    assert any(
        record.levelno == logging.WARNING
        and "Could not load churn dataset" in record.getMessage()
        and "ValueError" in record.getMessage()
        for record in records
    )
    assert any(
        record.levelno == logging.INFO
        and "No saved churn model" in record.getMessage()
        for record in records
    )


def test_successful_training_is_logged_after_model_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dataframe = pd.DataFrame({"placeholder": range(10)})
    split = (
        pd.DataFrame(index=range(8)),
        pd.DataFrame(index=range(2)),
        pd.Series([0, 1, 0, 1, 0, 1, 0, 1]),
        pd.Series([0, 1]),
    )
    pipeline = Pipeline(steps=[])
    metrics = TrainingMetrics(accuracy=0.8, f1=0.6, roc_auc=0.75)
    dataset = cast(ChurnDataset, TrainingDatasetStub(dataframe))
    app_stub = SimpleNamespace(state=SimpleNamespace(churn_model=None))
    model_path = tmp_path / "models" / "churn_model.joblib"
    history_path = tmp_path / "models" / "training_history.json"
    persistence_events: list[str] = []

    def record_model_save(_artifact: ChurnModelArtifact, _path: Path) -> None:
        persistence_events.append("model")

    def record_history_append(_entry: object, _path: Path) -> None:
        persistence_events.append("history")

    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    monkeypatch.setattr(main, "TRAINING_HISTORY_PATH", history_path)
    monkeypatch.setattr(main, "prepare_and_split", lambda _frame: split)
    monkeypatch.setattr(main, "train_churn_model", lambda *_args, **_kwargs: pipeline)
    monkeypatch.setattr(main, "evaluate_churn_model", lambda *_args: metrics)
    monkeypatch.setattr(main, "save_churn_model", record_model_save)
    monkeypatch.setattr(main, "append_training_entry", record_history_append)
    caplog.set_level(logging.INFO, logger=main.logger.name)

    response = main.train_model(
        request=make_request(app_stub),
        config=TrainingConfigChurn(model_type="logreg"),
        dataset=dataset,
    )

    assert response.accuracy == metrics.accuracy
    assert response.f1 == metrics.f1
    assert persistence_events == ["model", "history"]
    assert isinstance(app_stub.state.churn_model, ChurnModelArtifact)
    messages = logger_messages(caplog)
    assert any(
        "Starting churn model training" in message
        and "model_type=logreg" in message
        and "dataset_rows=10" in message
        and "train_rows=8" in message
        and "test_rows=2" in message
        for message in messages
    )
    assert any(
        "Completed churn model training" in message
        and "accuracy=0.800000" in message
        and "f1=0.600000" in message
        and str(model_path) in message
        for message in messages
    )


@pytest.mark.parametrize(
    ("batch_size", "input_type"),
    [(1, "single"), (2, "batch")],
)
def test_single_and_batch_prediction_calls_are_logged(
    monkeypatch: pytest.MonkeyPatch,
    valid_record: dict[str, object],
    caplog: pytest.LogCaptureFixture,
    batch_size: int,
    input_type: str,
) -> None:
    vector = make_feature_vector(valid_record)
    payload = vector if batch_size == 1 else [vector, vector]
    artifact = cast(
        ChurnModelArtifact,
        SimpleNamespace(model_type="logreg"),
    )
    prediction = PredictionResponseChurn(
        predicted_class=0,
        class_probabilities={"0": 0.8, "1": 0.2},
    )
    monkeypatch.setattr(
        main,
        "predict_churn_batch",
        lambda _artifact, vectors: [prediction for _ in vectors],
    )
    caplog.set_level(logging.INFO, logger=main.logger.name)

    main.predict_churn(payload, artifact)

    messages = logger_messages(caplog)
    assert any(
        "Processing /predict request" in message
        and f"input_type={input_type}" in message
        and f"customer_count={batch_size}" in message
        and "model_type=logreg" in message
        for message in messages
    )
    assert any(
        "Completed churn prediction" in message
        and f"prediction_count={batch_size}" in message
        for message in messages
    )


def test_prediction_failure_logs_traceback(
    monkeypatch: pytest.MonkeyPatch,
    valid_record: dict[str, object],
    caplog: pytest.LogCaptureFixture,
) -> None:
    vector = make_feature_vector(valid_record)
    artifact = cast(
        ChurnModelArtifact,
        SimpleNamespace(model_type="logreg"),
    )

    def fail_prediction(*_args: object, **_kwargs: object) -> None:
        raise ValueError("simulated prediction failure")

    monkeypatch.setattr(main, "predict_churn_batch", fail_prediction)
    caplog.set_level(logging.ERROR, logger=main.logger.name)

    with pytest.raises(PredictionError):
        main.predict_churn(vector, artifact)

    failure_records = [
        record
        for record in logger_records(caplog)
        if "Churn prediction failed" in record.getMessage()
    ]
    assert len(failure_records) == 1
    assert failure_records[0].levelno == logging.ERROR
    assert failure_records[0].exc_info is not None
    assert failure_records[0].exc_info[0] is ValueError


@pytest.mark.anyio
async def test_validation_error_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    request = cast(
        Request,
        SimpleNamespace(method="POST", url=SimpleNamespace(path="/predict")),
    )
    exception = cast(
        RequestValidationError,
        SimpleNamespace(
            errors=lambda: [
                {
                    "loc": ("body", "monthly_fee"),
                    "msg": "Field required",
                    "type": "missing",
                }
            ]
        ),
    )
    caplog.set_level(logging.WARNING, logger=main.logger.name)

    response = await main.request_validation_exception_handler(request, exception)

    assert response.status_code == 422
    assert any(
        record.levelno == logging.WARNING
        and "Request validation failed" in record.getMessage()
        and "status=422" in record.getMessage()
        and "code=request_validation_error" in record.getMessage()
        for record in logger_records(caplog)
    )


@pytest.mark.anyio
async def test_handled_http_error_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    request = cast(
        Request,
        SimpleNamespace(method="POST", url=SimpleNamespace(path="/predict")),
    )
    caplog.set_level(logging.WARNING, logger=main.logger.name)

    response = await main.http_exception_handler(request, ModelNotTrainedError())

    assert response.status_code == 503
    assert any(
        record.levelno == logging.WARNING
        and "HTTP error" in record.getMessage()
        and "status=503" in record.getMessage()
        and "code=model_not_trained" in record.getMessage()
        for record in logger_records(caplog)
    )
