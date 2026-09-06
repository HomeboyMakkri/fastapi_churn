from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import FastAPI, Request

from src import main
from src.dataset import ChurnDataset
from src.model_store import ChurnModelArtifact
from src.schemas import (
    FeatureVectorChurn,
    PredictionResponseChurn,
    TrainingConfigChurn,
)
from src.training_history import load_training_history


@pytest.mark.anyio
async def test_day11_training_history_survives_restart_and_supports_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_record: dict[str, object],
) -> None:
    model_path = tmp_path / "models" / "churn_model.joblib"
    history_path = tmp_path / "models" / "training_history.json"
    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    monkeypatch.setattr(main, "TRAINING_HISTORY_PATH", history_path)

    dataset = ChurnDataset(main.DATASET_PATH)
    dataset.load()
    app_stub = SimpleNamespace(state=SimpleNamespace(churn_model=None))
    request = cast(Request, SimpleNamespace(app=app_stub))
    feature_vector = FeatureVectorChurn.model_validate(
        {
            name: value
            for name, value in valid_record.items()
            if name != "churn"
        }
    )

    first_metrics = main.train_model(
        request=request,
        config=TrainingConfigChurn(
            model_type="logreg",
            hyperparameters={"C": 0.75},
        ),
        dataset=dataset,
    )
    first_artifact = app_stub.state.churn_model
    assert isinstance(first_artifact, ChurnModelArtifact)

    second_metrics = main.train_model(
        request=request,
        config=TrainingConfigChurn(
            model_type="random_forest",
            hyperparameters={"n_estimators": 10, "random_state": 7},
        ),
        dataset=dataset,
    )
    second_artifact = app_stub.state.churn_model
    assert isinstance(second_artifact, ChurnModelArtifact)

    persisted_history = load_training_history(history_path)
    assert [entry.model_type for entry in persisted_history] == [
        "logreg",
        "random_forest",
    ]
    assert persisted_history[0].metrics.accuracy == first_metrics.accuracy
    assert persisted_history[0].metrics.f1 == first_metrics.f1
    assert persisted_history[1].metrics.accuracy == second_metrics.accuracy
    assert persisted_history[1].metrics.f1 == second_metrics.f1
    assert all(0 <= entry.metrics.roc_auc <= 1 for entry in persisted_history)

    all_metrics = main.get_model_metrics(limit=10, model_type=None)
    assert all_metrics.latest == persisted_history[1]
    assert all_metrics.history == list(reversed(persisted_history))

    logreg_metrics = main.get_model_metrics(limit=1, model_type="logreg")
    assert logreg_metrics.latest == persisted_history[0]
    assert logreg_metrics.history == [persisted_history[0]]

    status_before_restart = main.get_model_status(request)
    assert status_before_restart.is_trained is True
    assert status_before_restart.model_type == "random_forest"
    assert status_before_restart.metrics == second_metrics

    prediction_before_restart = main.predict_churn(
        feature_vector,
        second_artifact,
    )
    assert isinstance(prediction_before_restart, PredictionResponseChurn)

    monkeypatch.setattr(main, "DATASET_PATH", tmp_path / "missing.csv")
    restarted_app = cast(FastAPI, app_stub)
    async with main.lifespan(restarted_app):
        restored_artifact = app_stub.state.churn_model
        assert app_stub.state.churn_dataset is None
        assert isinstance(restored_artifact, ChurnModelArtifact)

        status_after_restart = main.get_model_status(request)
        assert status_after_restart.is_trained is True
        assert status_after_restart.model_type == "random_forest"
        assert status_after_restart.metrics == second_metrics

        metrics_after_restart = main.get_model_metrics(
            limit=10,
            model_type=None,
        )
        assert metrics_after_restart == all_metrics

        prediction_after_restart = main.predict_churn(
            feature_vector,
            restored_artifact,
        )
        assert prediction_after_restart == prediction_before_restart
