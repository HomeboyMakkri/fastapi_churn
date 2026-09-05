from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException, Request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import main
from src.dataset import ChurnDataset
from src.errors import ModelConfigurationApiError
from src.model_store import (
    ChurnModelArtifact,
    ModelPersistenceError,
    load_churn_model,
)
from src.preprocessing import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from src.schemas import (
    FeatureVectorChurn,
    PredictionResponseChurn,
    TrainingConfigChurn,
)


def test_training_config_is_exposed_in_openapi() -> None:
    openapi = main.app.openapi()
    operation = openapi["paths"]["/model/train"]["post"]
    request_body = operation["requestBody"]
    request_schema = request_body["content"]["application/json"]["schema"]
    config_schema = openapi["components"]["schemas"]["TrainingConfigChurn"]

    assert request_body["required"] is True
    assert request_schema == {
        "$ref": "#/components/schemas/TrainingConfigChurn"
    }
    assert config_schema["additionalProperties"] is False
    assert config_schema["required"] == ["model_type"]
    assert config_schema["properties"]["model_type"]["enum"] == [
        "logreg",
        "random_forest",
    ]
    hyperparameter_types = {
        item["type"]
        for item in config_schema["properties"]["hyperparameters"][
            "additionalProperties"
        ]["anyOf"]
    }
    assert hyperparameter_types == {
        "string",
        "integer",
        "number",
        "boolean",
        "null",
    }


@pytest.mark.parametrize(
    "config_payload",
    [
        {
            "model_type": "logreg",
            "hyperparameters": {"C": 0.75, "max_iter": 250},
        },
        {
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": 10, "random_state": 7},
        },
    ],
)
def test_trained_and_restored_models_support_predict_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_record: dict[str, object],
    config_payload: dict[str, object],
) -> None:
    model_path = tmp_path / "churn_model.joblib"
    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    dataset = ChurnDataset(main.DATASET_PATH)
    dataset.load()
    app_stub = SimpleNamespace(state=SimpleNamespace(churn_model=None))
    request = cast(Request, SimpleNamespace(app=app_stub))
    config = TrainingConfigChurn.model_validate(config_payload)
    feature_vector = FeatureVectorChurn.model_validate(
        {key: value for key, value in valid_record.items() if key != "churn"}
    )

    main.train_model(request=request, config=config, dataset=dataset)

    trained_artifact = app_stub.state.churn_model
    restored_artifact = load_churn_model(model_path)
    assert isinstance(trained_artifact, ChurnModelArtifact)
    trained_prediction = main.predict_churn(feature_vector, trained_artifact)
    restored_prediction = main.predict_churn(feature_vector, restored_artifact)

    restored_pipeline = restored_artifact.pipeline
    assert list(restored_pipeline.named_steps) == ["preprocessing", "classifier"]
    restored_preprocessor = restored_pipeline.named_steps["preprocessing"]
    assert isinstance(restored_preprocessor, ColumnTransformer)
    assert isinstance(
        restored_preprocessor.named_transformers_["num"],
        StandardScaler,
    )
    restored_encoder = restored_preprocessor.named_transformers_["cat"]
    assert isinstance(restored_encoder, OneHotEncoder)
    assert restored_encoder.handle_unknown == "ignore"
    transformer_columns = {
        name: tuple(columns)
        for name, _, columns in restored_preprocessor.transformers_
        if name in {"num", "cat"}
    }
    assert transformer_columns == {
        "num": NUMERIC_FEATURES,
        "cat": CATEGORICAL_FEATURES,
    }

    inference_frame = pd.DataFrame.from_records(
        [feature_vector.model_dump()],
        columns=list(FEATURES),
    )
    np.testing.assert_array_equal(
        trained_artifact.pipeline.predict(inference_frame),
        restored_pipeline.predict(inference_frame),
    )
    np.testing.assert_allclose(
        trained_artifact.pipeline.predict_proba(inference_frame),
        restored_pipeline.predict_proba(inference_frame),
    )

    assert isinstance(trained_prediction, PredictionResponseChurn)
    assert isinstance(restored_prediction, PredictionResponseChurn)
    assert trained_prediction == restored_prediction
    assert trained_prediction.predicted_class in {0, 1}
    assert sum(
        trained_prediction.class_probabilities.values()
    ) == pytest.approx(1.0)


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
    config = TrainingConfigChurn(
        model_type="logreg",
        hyperparameters={"C": 0.75, "max_iter": 250},
    )

    metrics = main.train_model(request=request, config=config, dataset=dataset)

    in_memory_artifact = main.app.state.churn_model
    persisted_artifact = load_churn_model(model_path)
    ready_status = main.get_model_status(request)
    sample = dataset.dataframe.drop(columns="churn").iloc[[0]]
    assert isinstance(in_memory_artifact, ChurnModelArtifact)
    assert in_memory_artifact.trained_at.utcoffset() is not None
    assert persisted_artifact.accuracy == metrics.accuracy
    assert persisted_artifact.f1 == metrics.f1
    assert persisted_artifact.model_type == config.model_type
    assert persisted_artifact.hyperparameters == config.hyperparameters
    assert in_memory_artifact.model_type == config.model_type
    assert in_memory_artifact.hyperparameters == config.hyperparameters
    assert ready_status.is_trained is True
    assert ready_status.last_trained_at == persisted_artifact.trained_at
    assert ready_status.metrics == metrics
    assert ready_status.model_type == config.model_type
    assert ready_status.hyperparameters == config.hyperparameters

    monkeypatch.setattr(main, "DATASET_PATH", tmp_path / "missing.csv")
    main.app.state.churn_model = None
    async with main.lifespan(main.app):
        restored_artifact = main.app.state.churn_model

        assert main.app.state.churn_dataset is None
        assert isinstance(restored_artifact, ChurnModelArtifact)
        assert restored_artifact.trained_at == persisted_artifact.trained_at
        assert restored_artifact.model_type == config.model_type
        assert restored_artifact.hyperparameters == config.hyperparameters
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
        assert restored_status.model_type == config.model_type
        assert restored_status.hyperparameters == config.hyperparameters


def test_training_does_not_publish_model_when_persistence_fails(
    monkeypatch: pytest.MonkeyPatch,
    trained_artifact: ChurnModelArtifact,
) -> None:
    dataset = ChurnDataset(main.DATASET_PATH)
    dataset.load()
    app_stub = SimpleNamespace(
        state=SimpleNamespace(churn_model=trained_artifact)
    )
    request = cast(Request, SimpleNamespace(app=app_stub))
    config = TrainingConfigChurn(model_type="logreg")

    def fail_to_save(*args: object, **kwargs: object) -> None:
        raise ModelPersistenceError("simulated persistence failure")

    monkeypatch.setattr(main, "save_churn_model", fail_to_save)

    with pytest.raises(ModelPersistenceError, match="simulated persistence failure"):
        main.train_model(request=request, config=config, dataset=dataset)

    assert app_stub.state.churn_model is trained_artifact


def test_training_creates_and_persists_random_forest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "random_forest.joblib"
    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    dataset = ChurnDataset(main.DATASET_PATH)
    dataset.load()
    app_stub = SimpleNamespace(state=SimpleNamespace(churn_model=None))
    request = cast(Request, SimpleNamespace(app=app_stub))
    config = TrainingConfigChurn(
        model_type="random_forest",
        hyperparameters={
            "n_estimators": 10,
            "max_depth": 3,
            "random_state": 7,
        },
    )

    main.train_model(request=request, config=config, dataset=dataset)

    artifact = app_stub.state.churn_model
    persisted_artifact = load_churn_model(model_path)
    assert isinstance(artifact, ChurnModelArtifact)
    assert artifact.model_type == "random_forest"
    assert artifact.hyperparameters == config.hyperparameters
    classifier = artifact.pipeline.named_steps["classifier"]
    assert isinstance(classifier, RandomForestClassifier)
    assert classifier.n_estimators == 10
    assert classifier.max_depth == 3
    assert classifier.random_state == 7
    assert persisted_artifact.model_type == artifact.model_type
    assert persisted_artifact.hyperparameters == artifact.hyperparameters
    status = main.get_model_status(request)
    assert status.model_type == "random_forest"
    assert status.hyperparameters == config.hyperparameters


@pytest.mark.parametrize(
    "hyperparameters",
    [
        {"unknown_parameter": 1},
        {"C": 0.0},
    ],
)
def test_training_converts_configuration_errors_to_422(
    hyperparameters: dict[str, object],
) -> None:
    dataset = ChurnDataset(main.DATASET_PATH)
    dataset.load()
    app_stub = SimpleNamespace(state=SimpleNamespace(churn_model=None))
    request = cast(Request, SimpleNamespace(app=app_stub))
    config = TrainingConfigChurn(
        model_type="logreg",
        hyperparameters=hyperparameters,
    )

    with pytest.raises(HTTPException) as raised:
        main.train_model(request=request, config=config, dataset=dataset)

    assert raised.value.status_code == 422
    assert isinstance(raised.value, ModelConfigurationApiError)
    assert raised.value.code == "model_configuration_error"
    assert isinstance(raised.value.details, dict)
    assert "hyperparameters" in raised.value.details["reason"]
    assert app_stub.state.churn_model is None


def test_model_status_reports_untrained_state() -> None:
    main.app.state.churn_model = None
    request = cast(Request, SimpleNamespace(app=main.app))

    status = main.get_model_status(request)

    assert status.model_dump() == {
        "is_trained": False,
        "last_trained_at": None,
        "metrics": None,
        "model_type": None,
        "hyperparameters": None,
    }


def test_get_dataset_returns_503_when_dataset_is_unavailable() -> None:
    request = cast(
        Request,
        SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(churn_dataset=None))
        ),
    )

    with pytest.raises(HTTPException) as raised:
        main.get_dataset(request)

    assert raised.value.status_code == 503
    assert raised.value.detail == "Churn dataset is not available"


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
