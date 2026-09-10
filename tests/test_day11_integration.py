from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI, Request

from src.application import create_app
from src.config import AppSettings
from src.dataset import ChurnDataset
from src.lifespan import lifespan
from src.model_store import ChurnModelArtifact
from src.routers.model import get_model_metrics, get_model_status, train_model
from src.routers.prediction import predict_churn
from src.schemas import (
    FeatureVectorChurn,
    PredictionResponseChurn,
    TrainingConfigChurn,
)
from src.training_history import load_training_history


@pytest.mark.anyio
async def test_day11_training_history_survives_restart_and_supports_inference(
    tmp_path: Path,
    valid_record: dict[str, object],
) -> None:
    model_path = tmp_path / "models" / "churn_model.joblib"
    history_path = tmp_path / "models" / "training_history.json"
    settings = AppSettings(
        model_path=model_path,
        training_history_path=history_path,
    )
    app = create_app(settings)
    dataset = ChurnDataset(settings.dataset_path)
    dataset.load()
    request = cast(Request, type("RequestStub", (), {"app": app})())
    feature_vector = FeatureVectorChurn.model_validate(
        {
            name: value
            for name, value in valid_record.items()
            if name != "churn"
        }
    )

    first_metrics = train_model(
        request=request,
        config=TrainingConfigChurn(
            model_type="logreg",
            hyperparameters={"C": 0.75},
        ),
        dataset=dataset,
    )
    first_artifact = app.state.churn_model
    assert isinstance(first_artifact, ChurnModelArtifact)

    second_metrics = train_model(
        request=request,
        config=TrainingConfigChurn(
            model_type="random_forest",
            hyperparameters={"n_estimators": 10, "random_state": 7},
        ),
        dataset=dataset,
    )
    second_artifact = app.state.churn_model
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

    all_metrics = get_model_metrics(request, limit=10, model_type=None)
    assert all_metrics.latest == persisted_history[1]
    assert all_metrics.history == list(reversed(persisted_history))

    logreg_metrics = get_model_metrics(request, limit=1, model_type="logreg")
    assert logreg_metrics.latest == persisted_history[0]
    assert logreg_metrics.history == [persisted_history[0]]

    status_before_restart = get_model_status(request)
    assert status_before_restart.is_trained is True
    assert status_before_restart.model_type == "random_forest"
    assert status_before_restart.metrics == second_metrics

    prediction_before_restart = predict_churn(
        feature_vector,
        second_artifact,
    )
    assert isinstance(prediction_before_restart, PredictionResponseChurn)

    restarted_app = create_app(
        AppSettings(
            dataset_path=tmp_path / "missing.csv",
            model_path=model_path,
            training_history_path=history_path,
        )
    )
    restarted_request = cast(
        Request,
        type("RequestStub", (), {"app": restarted_app})(),
    )
    async with lifespan(restarted_app):
        restored_artifact = restarted_app.state.churn_model
        assert restarted_app.state.churn_dataset is None
        assert isinstance(restored_artifact, ChurnModelArtifact)

        status_after_restart = get_model_status(restarted_request)
        assert status_after_restart.is_trained is True
        assert status_after_restart.model_type == "random_forest"
        assert status_after_restart.metrics == second_metrics

        metrics_after_restart = get_model_metrics(
            restarted_request,
            limit=10,
            model_type=None,
        )
        assert metrics_after_restart == all_metrics

        prediction_after_restart = predict_churn(
            feature_vector,
            restored_artifact,
        )
        assert prediction_after_restart == prediction_before_restart
