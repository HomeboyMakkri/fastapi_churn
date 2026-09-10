"""Model schema, training, status, and metrics endpoints."""

from collections.abc import Callable
import logging
from pathlib import Path
from typing import cast

from fastapi import APIRouter, Query, Request

from ..dataset_contract import CHURN_DATASET_CONTRACT
from ..dependencies import DatasetDependency
from ..evaluation import evaluate_churn_model
from ..model import train_churn_model
from ..model_store import ChurnModelArtifact, save_churn_model
from ..openapi_examples import (
    DATASET_EMPTY_ERROR_EXAMPLE,
    INTERNAL_SERVER_ERROR_EXAMPLE,
    MODEL_CONFIGURATION_ERROR_EXAMPLE,
)
from ..preprocessing import prepare_and_split
from ..schemas import (
    ErrorResponse,
    FeatureGroup,
    FeatureVectorChurn,
    FeatureValueType,
    ModelFeatureSchemaChurn,
    ModelMetricsResponse,
    ModelSchemaChurn,
    ModelStatus,
    ModelTrainingInfo,
    ModelType,
    TrainingConfigChurn,
    TrainingHistoryEntry,
)
from ..services.training import (
    AppendHistoryFunction,
    EvaluateModelFunction,
    PrepareAndSplitFunction,
    SaveModelFunction,
    TrainModelFunction,
    train_and_persist_churn_model,
)
from ..training_history import append_training_entry, load_training_history


logger = logging.getLogger("uvicorn.error.src.main")
router = APIRouter()

LoadHistoryFunction = Callable[[Path], list[TrainingHistoryEntry]]
_DEFAULT_PREPARE_FUNCTION = prepare_and_split
_DEFAULT_TRAINING_FUNCTION = train_churn_model
_DEFAULT_EVALUATION_FUNCTION = evaluate_churn_model
_DEFAULT_SAVE_FUNCTION = save_churn_model
_DEFAULT_APPEND_HISTORY_FUNCTION = append_training_entry
_DEFAULT_LOAD_HISTORY_FUNCTION = load_training_history


def _legacy_or_local(
    legacy_value: object,
    local_value: object,
    default_value: object,
) -> object:
    """Honor temporary src.main monkeypatch seams during router migration."""
    return legacy_value if legacy_value is not default_value else local_value


def _get_feature_value_type(feature_name: str) -> FeatureValueType:
    feature_schema = FeatureVectorChurn.model_json_schema()["properties"][
        feature_name
    ]
    value_type = feature_schema.get("type")
    if value_type not in ("number", "integer", "string"):
        raise RuntimeError(
            f"Unsupported JSON type for feature {feature_name!r}: {value_type!r}"
        )

    return cast(FeatureValueType, value_type)


@router.get(
    "/model/schema",
    response_model=ModelSchemaChurn,
    summary="Get the churn model input schema",
    description=(
        "Returns the ordered churn feature contract, including each feature's "
        "expected JSON value type and preprocessing group. The schema is "
        "available even when no dataset or trained model is loaded."
    ),
)
def get_model_schema() -> ModelSchemaChurn:
    """Return the ordered model-input contract without requiring runtime state."""
    numeric_features = set(CHURN_DATASET_CONTRACT.numeric_features)
    features = [
        ModelFeatureSchemaChurn(
            name=feature_name,
            value_type=_get_feature_value_type(feature_name),
            group=cast(
                FeatureGroup,
                "numeric" if feature_name in numeric_features else "categorical",
            ),
        )
        for feature_name in CHURN_DATASET_CONTRACT.features
    ]
    return ModelSchemaChurn(features=features)


@router.post(
    "/model/train",
    response_model=ModelTrainingInfo,
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Invalid training configuration or dataset",
            "content": {
                "application/json": {
                    "example": MODEL_CONFIGURATION_ERROR_EXAMPLE,
                }
            },
        },
        503: {
            "model": ErrorResponse,
            "description": "The training dataset is unavailable or empty",
            "content": {
                "application/json": {
                    "example": DATASET_EMPTY_ERROR_EXAMPLE,
                }
            },
        },
        500: {
            "model": ErrorResponse,
            "description": "Training or model persistence failed",
            "content": {
                "application/json": {
                    "example": INTERNAL_SERVER_ERROR_EXAMPLE,
                }
            },
        },
    },
)
def train_model(
    request: Request,
    config: TrainingConfigChurn,
    dataset: DatasetDependency,
) -> ModelTrainingInfo:
    from .. import main as main_module

    result = train_and_persist_churn_model(
        dataset,
        config,
        model_path=main_module.MODEL_PATH,
        training_history_path=main_module.TRAINING_HISTORY_PATH,
        prepare_function=cast(
            PrepareAndSplitFunction,
            _legacy_or_local(
                main_module.prepare_and_split,
                prepare_and_split,
                _DEFAULT_PREPARE_FUNCTION,
            ),
        ),
        training_function=cast(
            TrainModelFunction,
            _legacy_or_local(
                main_module.train_churn_model,
                train_churn_model,
                _DEFAULT_TRAINING_FUNCTION,
            ),
        ),
        evaluation_function=cast(
            EvaluateModelFunction,
            _legacy_or_local(
                main_module.evaluate_churn_model,
                evaluate_churn_model,
                _DEFAULT_EVALUATION_FUNCTION,
            ),
        ),
        save_function=cast(
            SaveModelFunction,
            _legacy_or_local(
                main_module.save_churn_model,
                save_churn_model,
                _DEFAULT_SAVE_FUNCTION,
            ),
        ),
        append_history_function=cast(
            AppendHistoryFunction,
            _legacy_or_local(
                main_module.append_training_entry,
                append_training_entry,
                _DEFAULT_APPEND_HISTORY_FUNCTION,
            ),
        ),
    )
    request.app.state.churn_model = result.artifact
    response = ModelTrainingInfo(
        accuracy=result.metrics.accuracy,
        f1=result.metrics.f1,
    )
    logger.info(
        "Completed churn model training: model_type=%s accuracy=%.6f "
        "f1=%.6f model_path=%s",
        config.model_type,
        response.accuracy,
        response.f1,
        main_module.MODEL_PATH,
    )
    return response


@router.get("/model/status", response_model=ModelStatus)
def get_model_status(request: Request) -> ModelStatus:
    artifact = getattr(request.app.state, "churn_model", None)
    if artifact is None:
        return ModelStatus(
            is_trained=False,
            last_trained_at=None,
            metrics=None,
            model_type=None,
            hyperparameters=None,
        )

    artifact = cast(ChurnModelArtifact, artifact)
    return ModelStatus(
        is_trained=True,
        last_trained_at=artifact.trained_at,
        metrics=ModelTrainingInfo(
            accuracy=artifact.accuracy,
            f1=artifact.f1,
        ),
        model_type=artifact.model_type,
        hyperparameters=artifact.hyperparameters,
    )


def _get_model_metrics(
    limit: int,
    model_type: ModelType | None,
    *,
    training_history_path: Path,
    load_history_function: LoadHistoryFunction,
) -> ModelMetricsResponse:
    history = load_history_function(training_history_path)
    if model_type is not None:
        history = [
            entry for entry in history if entry.model_type == model_type
        ]

    newest_first = list(reversed(history))
    return ModelMetricsResponse(
        latest=newest_first[0] if newest_first else None,
        history=newest_first[:limit],
    )


@router.get(
    "/model/metrics",
    response_model=ModelMetricsResponse,
    summary="Get churn model training metrics",
    description=(
        "Returns the latest matching training record and a bounded training "
        "history ordered from newest to oldest. History can be filtered by "
        "model type."
    ),
    responses={
        500: {
            "model": ErrorResponse,
            "description": "Training history could not be loaded",
            "content": {
                "application/json": {
                    "example": INTERNAL_SERVER_ERROR_EXAMPLE,
                }
            },
        },
    },
)
def get_model_metrics(
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of training records to return",
    ),
    model_type: ModelType | None = Query(
        default=None,
        description="Return only records for this classifier type",
    ),
) -> ModelMetricsResponse:
    from .. import main as main_module

    return _get_model_metrics(
        limit,
        model_type,
        training_history_path=main_module.TRAINING_HISTORY_PATH,
        load_history_function=cast(
            LoadHistoryFunction,
            _legacy_or_local(
                main_module.load_training_history,
                load_training_history,
                _DEFAULT_LOAD_HISTORY_FUNCTION,
            ),
        ),
    )
