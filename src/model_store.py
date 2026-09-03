"""Persist and restore trained churn-model artifacts."""

from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Literal

import joblib
from sklearn.pipeline import Pipeline


class ModelPersistenceError(RuntimeError):
    """Raised when a model artifact cannot be serialized or restored."""


ModelType = Literal["logreg", "random_forest"]
HyperparameterValue = str | int | float | bool | None


@dataclass(frozen=True)
class ChurnModelArtifact:
    """A trained pipeline and the metadata needed after an app restart."""

    pipeline: Pipeline
    trained_at: datetime
    accuracy: float
    f1: float
    model_type: ModelType = "logreg"
    hyperparameters: dict[str, HyperparameterValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline, Pipeline):
            raise TypeError("pipeline must be a scikit-learn Pipeline")
        if not isinstance(self.trained_at, datetime):
            raise TypeError("trained_at must be a datetime")
        if self.trained_at.tzinfo is None or self.trained_at.utcoffset() is None:
            raise ValueError("trained_at must include timezone information")
        for name, value in (("accuracy", self.accuracy), ("f1", self.f1)):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.model_type not in ("logreg", "random_forest"):
            raise ValueError("model_type must be 'logreg' or 'random_forest'")
        if not isinstance(self.hyperparameters, dict):
            raise TypeError("hyperparameters must be a dictionary")

        hyperparameters = self.hyperparameters.copy()
        for name, value in hyperparameters.items():
            if not isinstance(name, str):
                raise TypeError("hyperparameter names must be strings")
            if value is not None and type(value) not in (str, int, float, bool):
                raise TypeError(
                    "hyperparameter values must be JSON-compatible scalars"
                )
            if isinstance(value, float) and not isfinite(value):
                raise ValueError("float hyperparameter values must be finite")

        object.__setattr__(self, "hyperparameters", hyperparameters)


def save_churn_model(artifact: ChurnModelArtifact, path: Path) -> None:
    """Atomically save a churn-model artifact to a joblib file."""
    if not isinstance(artifact, ChurnModelArtifact):
        raise TypeError("artifact must be a ChurnModelArtifact")

    _validate_model_suffix(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")

    try:
        joblib.dump(artifact, temporary_path)
        temporary_path.replace(path)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise ModelPersistenceError(f"Could not save model artifact to {path}") from error


def load_churn_model(path: Path) -> ChurnModelArtifact:
    """Load and validate a churn-model artifact from a joblib file."""
    _validate_model_suffix(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Model path is not a file: {path}")

    try:
        artifact = joblib.load(path)
    except Exception as error:
        raise ModelPersistenceError(f"Could not load model artifact from {path}") from error

    if not isinstance(artifact, ChurnModelArtifact):
        raise ModelPersistenceError(
            f"File does not contain a ChurnModelArtifact: {path}"
        )

    try:
        return ChurnModelArtifact(
            pipeline=artifact.pipeline,
            trained_at=artifact.trained_at,
            accuracy=artifact.accuracy,
            f1=artifact.f1,
            model_type=artifact.model_type,
            hyperparameters=artifact.hyperparameters,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ModelPersistenceError(
            f"File contains an invalid ChurnModelArtifact: {path}"
        ) from error


def _validate_model_suffix(path: Path) -> None:
    if path.suffix.lower() != ".joblib":
        raise ValueError(f"Model file must use the .joblib suffix: {path}")
