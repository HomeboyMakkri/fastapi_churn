from datetime import datetime, timezone
from pathlib import Path

import joblib
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.model_store import (
    ChurnModelArtifact,
    ModelPersistenceError,
    load_churn_model,
    save_churn_model,
)


def make_artifact() -> ChurnModelArtifact:
    pipeline = Pipeline(
        steps=[("classifier", DummyClassifier(strategy="most_frequent"))]
    )
    pipeline.fit([[0], [1]], [0, 1])
    return ChurnModelArtifact(
        pipeline=pipeline,
        trained_at=datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc),
        accuracy=0.8,
        f1=0.6,
        model_type="random_forest",
        hyperparameters={"n_estimators": 50, "random_state": 7},
    )


def test_save_and_load_churn_model_round_trip(tmp_path: Path) -> None:
    artifact = make_artifact()
    model_path = tmp_path / "nested" / "churn_model.joblib"

    save_churn_model(artifact, model_path)
    restored = load_churn_model(model_path)

    assert model_path.is_file()
    assert restored.trained_at == artifact.trained_at
    assert restored.accuracy == artifact.accuracy
    assert restored.f1 == artifact.f1
    assert restored.model_type == artifact.model_type
    assert restored.hyperparameters == artifact.hyperparameters
    assert restored.pipeline.predict([[2]]).tolist() == [0]


def test_artifact_copies_hyperparameters() -> None:
    hyperparameters = {"n_estimators": 50, "random_state": 7}
    artifact = make_artifact()
    values = {
        "pipeline": artifact.pipeline,
        "trained_at": artifact.trained_at,
        "accuracy": artifact.accuracy,
        "f1": artifact.f1,
        "model_type": "random_forest",
        "hyperparameters": hyperparameters,
    }

    copied_artifact = ChurnModelArtifact(**values)  # type: ignore[arg-type]
    hyperparameters["n_estimators"] = 100

    assert copied_artifact.hyperparameters == {
        "n_estimators": 50,
        "random_state": 7,
    }


def test_load_churn_model_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        load_churn_model(tmp_path / "missing.joblib")


def test_load_churn_model_rejects_unexpected_object(tmp_path: Path) -> None:
    model_path = tmp_path / "churn_model.joblib"
    joblib.dump({"pipeline": "not an artifact"}, model_path)

    with pytest.raises(ModelPersistenceError, match="ChurnModelArtifact"):
        load_churn_model(model_path)


def test_load_churn_model_rejects_corrupted_file(tmp_path: Path) -> None:
    model_path = tmp_path / "churn_model.joblib"
    model_path.write_bytes(b"not a joblib artifact")

    with pytest.raises(ModelPersistenceError, match="Could not load"):
        load_churn_model(model_path)


def test_load_churn_model_rejects_invalid_artifact(tmp_path: Path) -> None:
    artifact = make_artifact()
    object.__setattr__(artifact, "model_type", "svm")
    model_path = tmp_path / "churn_model.joblib"
    joblib.dump(artifact, model_path)

    with pytest.raises(ModelPersistenceError, match="invalid ChurnModelArtifact"):
        load_churn_model(model_path)


def test_load_churn_model_rejects_artifact_with_missing_metadata(
    tmp_path: Path,
) -> None:
    artifact = make_artifact()
    object.__delattr__(artifact, "hyperparameters")
    model_path = tmp_path / "churn_model.joblib"
    joblib.dump(artifact, model_path)

    with pytest.raises(ModelPersistenceError, match="invalid ChurnModelArtifact"):
        load_churn_model(model_path)


@pytest.mark.parametrize(
    ("field", "value", "exception_type", "message"),
    [
        (
            "trained_at",
            datetime(2026, 9, 1, 12, 0),
            ValueError,
            "timezone",
        ),
        ("accuracy", 1.1, ValueError, "accuracy"),
        ("f1", -0.1, ValueError, "f1"),
        ("model_type", "svm", ValueError, "model_type"),
        ("hyperparameters", [], TypeError, "dictionary"),
        (
            "hyperparameters",
            {"layers": [10, 5]},
            TypeError,
            "JSON-compatible scalars",
        ),
        (
            "hyperparameters",
            {1: "invalid name"},
            TypeError,
            "names must be strings",
        ),
        (
            "hyperparameters",
            {"threshold": float("nan")},
            ValueError,
            "finite",
        ),
    ],
)
def test_artifact_rejects_invalid_metadata(
    field: str,
    value: object,
    exception_type: type[Exception],
    message: str,
) -> None:
    artifact = make_artifact()
    values: dict[str, object] = {
        "pipeline": artifact.pipeline,
        "trained_at": artifact.trained_at,
        "accuracy": artifact.accuracy,
        "f1": artifact.f1,
        "model_type": artifact.model_type,
        "hyperparameters": artifact.hyperparameters,
    }
    values[field] = value

    with pytest.raises(exception_type, match=message):
        ChurnModelArtifact(**values)  # type: ignore[arg-type]
