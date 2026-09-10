from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.dataset import ChurnDataset
from src.errors import (
    DataPreparationError,
    DatasetEmptyError,
    DatasetUnavailableError,
    ModelConfigurationApiError,
)
from src.model import ModelConfigurationError
from src.model_store import ChurnModelArtifact
from src.schemas import (
    TrainingConfigChurn,
    TrainingHistoryEntry,
    TrainingMetrics,
)
from src.services.training import train_and_persist_churn_model


class DatasetStub:
    def __init__(self, dataframe: pd.DataFrame | None) -> None:
        self._dataframe = dataframe

    @property
    def dataframe(self) -> pd.DataFrame:
        if self._dataframe is None:
            raise RuntimeError("Dataset is not loaded")
        return self._dataframe.copy(deep=True)


def make_dataset(dataframe: pd.DataFrame | None) -> ChurnDataset:
    return cast(ChurnDataset, DatasetStub(dataframe))


def make_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return (
        pd.DataFrame({"feature": [1, 2]}),
        pd.DataFrame({"feature": [3]}),
        pd.Series([0, 1]),
        pd.Series([1]),
    )


def test_training_service_orchestrates_and_persists_in_order(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame({"placeholder": [1, 2, 3]})
    dataset = make_dataset(dataframe)
    config = TrainingConfigChurn(
        model_type="logreg",
        hyperparameters={"C": 0.5},
    )
    split = make_split()
    pipeline = Pipeline(steps=[])
    metrics = TrainingMetrics(accuracy=0.8, f1=0.6, roc_auc=0.75)
    model_path = tmp_path / "model.joblib"
    history_path = tmp_path / "history.json"
    events: list[str] = []
    saved_artifact: ChurnModelArtifact | None = None
    saved_entry: TrainingHistoryEntry | None = None

    def prepare_function(frame: pd.DataFrame):
        events.append("prepare")
        pd.testing.assert_frame_equal(frame, dataframe)
        return split

    def training_function(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs: object,
    ) -> Pipeline:
        events.append("train")
        assert X_train is split[0]
        assert y_train is split[2]
        assert kwargs == {
            "model_type": "logreg",
            "hyperparameters": {"C": 0.5},
        }
        return pipeline

    def evaluation_function(
        trained_pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> TrainingMetrics:
        events.append("evaluate")
        assert trained_pipeline is pipeline
        assert X_test is split[1]
        assert y_test is split[3]
        return metrics

    def save_function(artifact: ChurnModelArtifact, path: Path) -> None:
        nonlocal saved_artifact
        events.append("model")
        saved_artifact = artifact
        assert path == model_path

    def append_history_function(
        entry: TrainingHistoryEntry,
        path: Path,
    ) -> None:
        nonlocal saved_entry
        events.append("history")
        saved_entry = entry
        assert path == history_path

    result = train_and_persist_churn_model(
        dataset,
        config,
        model_path=model_path,
        training_history_path=history_path,
        prepare_function=prepare_function,
        training_function=training_function,
        evaluation_function=evaluation_function,
        save_function=save_function,
        append_history_function=append_history_function,
    )

    assert events == ["prepare", "train", "evaluate", "model", "history"]
    assert result.artifact is saved_artifact
    assert result.metrics == metrics
    assert result.artifact.pipeline is pipeline
    assert result.artifact.trained_at.utcoffset() is not None
    assert result.artifact.model_type == "logreg"
    assert result.artifact.hyperparameters == {"C": 0.5}
    assert saved_entry is not None
    assert saved_entry.trained_at == result.artifact.trained_at
    assert saved_entry.model_type == result.artifact.model_type
    assert saved_entry.hyperparameters == result.artifact.hyperparameters
    assert saved_entry.metrics == metrics


def test_training_service_saves_model_before_history(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    def save_function(_artifact: ChurnModelArtifact, _path: Path) -> None:
        events.append("model")

    def fail_history(_entry: TrainingHistoryEntry, _path: Path) -> None:
        events.append("history")
        raise RuntimeError("history failed")

    with pytest.raises(RuntimeError, match="history failed"):
        train_and_persist_churn_model(
            make_dataset(pd.DataFrame({"placeholder": [1]})),
            TrainingConfigChurn(model_type="logreg"),
            model_path=tmp_path / "model.joblib",
            training_history_path=tmp_path / "history.json",
            prepare_function=lambda _frame: make_split(),
            training_function=lambda *_args, **_kwargs: Pipeline(steps=[]),
            evaluation_function=lambda *_args: TrainingMetrics(
                accuracy=0.8,
                f1=0.6,
                roc_auc=0.75,
            ),
            save_function=save_function,
            append_history_function=fail_history,
        )

    assert events == ["model", "history"]


def test_training_service_rejects_unavailable_dataset(tmp_path: Path) -> None:
    with pytest.raises(DatasetUnavailableError) as raised:
        train_and_persist_churn_model(
            make_dataset(None),
            TrainingConfigChurn(model_type="logreg"),
            model_path=tmp_path / "model.joblib",
            training_history_path=tmp_path / "history.json",
        )

    assert raised.value.message == "Churn dataset is not loaded"


def test_training_service_rejects_empty_dataset(tmp_path: Path) -> None:
    with pytest.raises(DatasetEmptyError):
        train_and_persist_churn_model(
            make_dataset(pd.DataFrame()),
            TrainingConfigChurn(model_type="logreg"),
            model_path=tmp_path / "model.joblib",
            training_history_path=tmp_path / "history.json",
        )


def test_training_service_translates_data_preparation_error(
    tmp_path: Path,
) -> None:
    def fail_preparation(_frame: pd.DataFrame):
        raise ValueError("invalid training columns")

    with pytest.raises(DataPreparationError) as raised:
        train_and_persist_churn_model(
            make_dataset(pd.DataFrame({"placeholder": [1]})),
            TrainingConfigChurn(model_type="logreg"),
            model_path=tmp_path / "model.joblib",
            training_history_path=tmp_path / "history.json",
            prepare_function=fail_preparation,
        )

    assert raised.value.details == {"reason": "invalid training columns"}


def test_training_service_translates_model_configuration_error(
    tmp_path: Path,
) -> None:
    def fail_training(*_args: object, **_kwargs: object):
        raise ModelConfigurationError("unsupported hyperparameters")

    with pytest.raises(ModelConfigurationApiError) as raised:
        train_and_persist_churn_model(
            make_dataset(pd.DataFrame({"placeholder": [1]})),
            TrainingConfigChurn(model_type="logreg"),
            model_path=tmp_path / "model.joblib",
            training_history_path=tmp_path / "history.json",
            prepare_function=lambda _frame: make_split(),
            training_function=fail_training,
        )

    assert raised.value.details == {"reason": "unsupported hyperparameters"}
