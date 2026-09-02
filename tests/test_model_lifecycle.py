from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from fastapi import HTTPException, Request

from src import main
from src.dataset import ChurnDataset
from src.model_store import ChurnModelArtifact, load_churn_model


@pytest.mark.anyio
async def test_training_persists_model_and_next_lifespan_restores_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "models" / "churn_model.joblib"
    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    dataset = ChurnDataset(main.DATASET_PATH)
    dataset.load()
    request = cast(Request, SimpleNamespace(app=main.app))

    metrics = main.train_model(request=request, dataset=dataset)

    in_memory_artifact = main.app.state.churn_model
    persisted_artifact = load_churn_model(model_path)
    ready_status = main.get_model_status(request)
    sample = dataset.dataframe.drop(columns="churn").iloc[[0]]
    assert isinstance(in_memory_artifact, ChurnModelArtifact)
    assert in_memory_artifact.trained_at.utcoffset() is not None
    assert persisted_artifact.accuracy == metrics.accuracy
    assert persisted_artifact.f1 == metrics.f1
    assert ready_status.is_trained is True
    assert ready_status.last_trained_at == persisted_artifact.trained_at
    assert ready_status.metrics == metrics

    monkeypatch.setattr(main, "DATASET_PATH", tmp_path / "missing.csv")
    main.app.state.churn_model = None
    async with main.lifespan(main.app):
        restored_artifact = main.app.state.churn_model

        assert main.app.state.churn_dataset is None
        assert isinstance(restored_artifact, ChurnModelArtifact)
        assert restored_artifact.trained_at == persisted_artifact.trained_at
        restored_predictions = cast(
            np.ndarray,
            restored_artifact.pipeline.predict(sample),
        )
        persisted_predictions = cast(
            np.ndarray,
            persisted_artifact.pipeline.predict(sample),
        )
        assert restored_predictions.tolist() == persisted_predictions.tolist()
        restored_status = main.get_model_status(request)
        assert restored_status.is_trained is True
        assert restored_status.last_trained_at == persisted_artifact.trained_at
        assert restored_status.metrics == metrics


def test_model_status_reports_untrained_state() -> None:
    main.app.state.churn_model = None
    request = cast(Request, SimpleNamespace(app=main.app))

    status = main.get_model_status(request)

    assert status.model_dump() == {
        "is_trained": False,
        "last_trained_at": None,
        "metrics": None,
    }


def test_get_churn_model_returns_available_artifact() -> None:
    artifact = cast(ChurnModelArtifact, object())
    request = cast(
        Request,
        SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(churn_model=artifact))
        ),
    )

    assert main.get_churn_model(request) is artifact


def test_get_churn_model_returns_503_when_model_is_unavailable() -> None:
    request = cast(
        Request,
        SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(churn_model=None))
        ),
    )

    with pytest.raises(HTTPException) as raised:
        main.get_churn_model(request)

    assert raised.value.status_code == 503
    assert raised.value.detail == "Churn model is not trained"
