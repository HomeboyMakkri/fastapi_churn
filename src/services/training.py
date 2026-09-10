"""Orchestrate churn-model training and persistence."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from ..dataset import ChurnDataset
from ..errors import (
    DataPreparationError,
    DatasetEmptyError,
    DatasetUnavailableError,
    ModelConfigurationApiError,
)
from ..evaluation import evaluate_churn_model
from ..model import ModelConfigurationError, train_churn_model
from ..model_store import ChurnModelArtifact, save_churn_model
from ..preprocessing import DatasetSplit, prepare_and_split
from ..schemas import (
    TrainingConfigChurn,
    TrainingHistoryEntry,
    TrainingMetrics,
)
from ..training_history import append_training_entry


logger = logging.getLogger("uvicorn.error.src.main")

PrepareAndSplitFunction = Callable[[pd.DataFrame], DatasetSplit]
TrainModelFunction = Callable[..., Pipeline]
EvaluateModelFunction = Callable[
    [Pipeline, pd.DataFrame, pd.Series],
    TrainingMetrics,
]
SaveModelFunction = Callable[[ChurnModelArtifact, Path], None]
AppendHistoryFunction = Callable[[TrainingHistoryEntry, Path], None]


@dataclass(frozen=True)
class TrainingResult:
    """Artifact and metrics produced by successful training."""

    artifact: ChurnModelArtifact
    metrics: TrainingMetrics


def train_and_persist_churn_model(
    dataset: ChurnDataset,
    config: TrainingConfigChurn,
    *,
    model_path: Path,
    training_history_path: Path,
    prepare_function: PrepareAndSplitFunction = prepare_and_split,
    training_function: TrainModelFunction = train_churn_model,
    evaluation_function: EvaluateModelFunction = evaluate_churn_model,
    save_function: SaveModelFunction = save_churn_model,
    append_history_function: AppendHistoryFunction = append_training_entry,
) -> TrainingResult:
    """Train, evaluate, and persist a churn model without publishing app state.

    Persistence is deliberately ordered model first, history second. This retains
    the existing recovery semantics: a history failure can leave the new artifact
    on disk, but callers must publish it to runtime state only after this function
    returns successfully.
    """
    try:
        dataframe = dataset.dataframe
    except RuntimeError as error:
        raise DatasetUnavailableError("Churn dataset is not loaded") from error

    if dataframe.empty:
        raise DatasetEmptyError

    try:
        X_train, X_test, y_train, y_test = prepare_function(dataframe)
    except (TypeError, ValueError) as error:
        raise DataPreparationError(str(error)) from error

    logger.info(
        "Starting churn model training: model_type=%s dataset_rows=%d "
        "train_rows=%d test_rows=%d",
        config.model_type,
        len(dataframe),
        len(X_train),
        len(X_test),
    )
    try:
        pipeline = training_function(
            X_train,
            y_train,
            model_type=config.model_type,
            hyperparameters=config.hyperparameters,
        )
    except ModelConfigurationError as error:
        raise ModelConfigurationApiError(str(error)) from error

    metrics = evaluation_function(pipeline, X_test, y_test)
    trained_at = datetime.now(timezone.utc)
    artifact = ChurnModelArtifact(
        pipeline=pipeline,
        trained_at=trained_at,
        accuracy=metrics.accuracy,
        f1=metrics.f1,
        model_type=config.model_type,
        hyperparameters=config.hyperparameters,
    )
    history_entry = TrainingHistoryEntry(
        trained_at=trained_at,
        model_type=config.model_type,
        hyperparameters=config.hyperparameters,
        metrics=metrics,
    )

    save_function(artifact, model_path)
    append_history_function(history_entry, training_history_path)

    return TrainingResult(
        artifact=artifact,
        metrics=metrics,
    )
