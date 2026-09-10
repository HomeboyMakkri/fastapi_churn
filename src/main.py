from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Annotated, cast

from fastapi import Body, FastAPI, Query, Request

from .dataset import ChurnDataset
from .dataset_contract import CHURN_DATASET_CONTRACT
from .dependencies import (
    DatasetDependency,
    ModelDependency,
    PreviewCount,
    get_churn_model,
    get_dataset,
)
from .evaluation import evaluate_churn_model
from .errors import (
    DataPreparationError,
    DatasetEmptyError,
    DatasetUnavailableError,
    ModelConfigurationApiError,
    PredictionError,
)
from .exception_handlers import (
    http_exception_handler,
    register_exception_handlers,
    request_validation_exception_handler,
    unhandled_exception_handler,
)
from .model import ModelConfigurationError, train_churn_model
from .model_store import (
    ChurnModelArtifact,
    ModelPersistenceError,
    load_churn_model,
    save_churn_model,
)
from .openapi_examples import (
    DATASET_EMPTY_ERROR_EXAMPLE,
    INTERNAL_SERVER_ERROR_EXAMPLE,
    MODEL_CONFIGURATION_ERROR_EXAMPLE,
    MODEL_NOT_TRAINED_ERROR_EXAMPLE,
    PREDICTION_FAILED_ERROR_EXAMPLE,
    PREDICTION_REQUEST_EXAMPLES,
    PREDICTION_RESPONSE_EXAMPLES,
    PREDICTION_VALIDATION_ERROR_EXAMPLE,
)
from .prediction import predict_churn_batch
from .preprocessing import (
    get_class_distribution,
    get_class_percentage,
    prepare_and_split,
)
from .schemas import (
    DatasetInfo,
    DatasetRowChurn,
    DatasetSplitInfo,
    ErrorResponse,
    FeatureGroup,
    FeatureVectorChurn,
    FeatureValueType,
    HealthStatus,
    ModelFeatureSchemaChurn,
    ModelMetricsResponse,
    ModelSchemaChurn,
    ModelStatus,
    ModelTrainingInfo,
    ModelType,
    PredictionPayload,
    PredictionResponseChurn,
    PredictionResult,
    TrainingConfigChurn,
    TrainingHistoryEntry,
)
from .training_history import append_training_entry, load_training_history


logger = logging.getLogger("uvicorn.error").getChild(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "churn_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
TRAINING_HISTORY_PATH = PROJECT_ROOT / "models" / "training_history.json"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.churn_dataset = None
    app.state.churn_model = None

    dataset = ChurnDataset(DATASET_PATH)
    logger.info("Loading churn dataset from %s", DATASET_PATH)
    try:
        dataset.load()
    except (OSError, ValueError) as error:
        logger.warning(
            "Could not load churn dataset from %s: error_type=%s",
            DATASET_PATH,
            type(error).__name__,
        )
    else:
        app.state.churn_dataset = dataset
        logger.info(
            "Loaded churn dataset from %s with %d rows",
            DATASET_PATH,
            len(dataset.dataframe),
        )

    logger.info("Restoring churn model from %s", MODEL_PATH)
    try:
        app.state.churn_model = load_churn_model(MODEL_PATH)
    except FileNotFoundError:
        logger.info("No saved churn model found at %s", MODEL_PATH)
    except (OSError, ValueError, ModelPersistenceError) as error:
        logger.warning(
            "Could not restore churn model from %s: error_type=%s",
            MODEL_PATH,
            type(error).__name__,
        )
    else:
        artifact = cast(ChurnModelArtifact, app.state.churn_model)
        logger.info(
            "Restored %s churn model trained at %s",
            artifact.model_type,
            artifact.trained_at.isoformat(),
        )

    yield


app = FastAPI(
    title="ML Churn Server",
    description="A FastAPI server for churn prediction and dataset management",
    version="1.0.0",
    lifespan=lifespan,
)
register_exception_handlers(app)


@app.get("/")
def read_root():
    return {"message": "ml churn server is running"}


@app.get(
    "/health",
    response_model=HealthStatus,
    summary="Get churn service health",
    description=(
        "Returns whether the churn dataset and trained model are currently "
        "available. The endpoint remains available when either resource is missing."
    ),
)
def get_health(request: Request) -> HealthStatus:
    """Return the availability of runtime resources without requiring them."""
    return HealthStatus(
        model_available=getattr(request.app.state, "churn_model", None) is not None,
        dataset_loaded=getattr(request.app.state, "churn_dataset", None) is not None,
    )


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


@app.get(
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


PredictionRequest = Annotated[
    PredictionPayload,
    Body(openapi_examples=PREDICTION_REQUEST_EXAMPLES),
]


@app.post(
    "/predict",
    response_model=PredictionResult,
    summary="Predict customer churn",
    description=(
        "Accepts one customer or a non-empty list of customers. "
        "Batch predictions are returned in the same order as the request."
    ),
    responses={
        200: {
            "description": "Churn class and probabilities for each customer",
            "content": {
                "application/json": {
                    "examples": PREDICTION_RESPONSE_EXAMPLES,
                }
            },
        },
        503: {
            "model": ErrorResponse,
            "description": "A trained churn model is not available",
            "content": {
                "application/json": {
                    "example": MODEL_NOT_TRAINED_ERROR_EXAMPLE,
                }
            },
        },
        422: {
            "model": ErrorResponse,
            "description": "Invalid feature set or value types",
            "content": {
                "application/json": {
                    "example": PREDICTION_VALIDATION_ERROR_EXAMPLE,
                }
            },
        },
        500: {
            "model": ErrorResponse,
            "description": "The model could not calculate a prediction",
            "content": {
                "application/json": {
                    "example": PREDICTION_FAILED_ERROR_EXAMPLE,
                }
            },
        },
    },
)
def predict_churn(
    payload: PredictionRequest,
    artifact: ModelDependency,
) -> PredictionResult:
    is_single_customer = isinstance(payload, FeatureVectorChurn)
    feature_vectors = [payload] if is_single_customer else payload
    input_type = "single" if is_single_customer else "batch"
    model_type = getattr(artifact, "model_type", "unknown")
    logger.info(
        "Processing /predict request: input_type=%s customer_count=%d model_type=%s",
        input_type,
        len(feature_vectors),
        model_type,
    )
    try:
        predictions = predict_churn_batch(artifact, feature_vectors)
    except Exception as error:
        logger.exception(
            "Churn prediction failed: input_type=%s customer_count=%d "
            "model_type=%s",
            input_type,
            len(feature_vectors),
            model_type,
        )
        raise PredictionError from error

    logger.info(
        "Completed churn prediction: prediction_count=%d",
        len(predictions),
    )
    if is_single_customer:
        return predictions[0]
    return predictions


@app.get("/dataset/preview", response_model=list[DatasetRowChurn])
def preview_dataset(
    dataset: DatasetDependency,
    count: PreviewCount = 5,
) -> list[DatasetRowChurn]:
    return dataset.to_rows()[:count]


@app.get("/dataset/info", response_model=DatasetInfo)
def get_dataset_info(dataset: DatasetDependency) -> DatasetInfo:
    dataframe = dataset.dataframe
    total_rows, total_columns = dataframe.shape
    target = dataframe["churn"]

    return DatasetInfo(
        total_rows=total_rows,
        total_columns=total_columns,
        column_names=dataframe.columns.tolist(),
        churn_distribution=get_class_distribution(target),
        churn_percentage=get_class_percentage(target),
    )


@app.get("/dataset/split-info", response_model=DatasetSplitInfo)
def get_dataset_split_info(
    dataset: DatasetDependency,
) -> DatasetSplitInfo:
    X_train, X_test, y_train, y_test = prepare_and_split(dataset.dataframe)

    return DatasetSplitInfo(
        train_rows=len(X_train),
        test_rows=len(X_test),
        feature_count=X_train.shape[1],
        train_churn_distribution=get_class_distribution(y_train),
        test_churn_distribution=get_class_distribution(y_test),
        train_churn_percentage=get_class_percentage(y_train),
        test_churn_percentage=get_class_percentage(y_test),
    )


@app.post(
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
    try:
        dataframe = dataset.dataframe
    except RuntimeError as error:
        raise DatasetUnavailableError("Churn dataset is not loaded") from error

    if dataframe.empty:
        raise DatasetEmptyError

    try:
        X_train, X_test, y_train, y_test = prepare_and_split(dataframe)
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
        pipeline = train_churn_model(
            X_train,
            y_train,
            model_type=config.model_type,
            hyperparameters=config.hyperparameters,
        )
    except ModelConfigurationError as error:
        raise ModelConfigurationApiError(str(error)) from error

    metrics = evaluate_churn_model(pipeline, X_test, y_test)

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

    save_churn_model(artifact, MODEL_PATH)
    append_training_entry(history_entry, TRAINING_HISTORY_PATH)
    request.app.state.churn_model = artifact
    logger.info(
        "Completed churn model training: model_type=%s accuracy=%.6f "
        "f1=%.6f model_path=%s",
        config.model_type,
        metrics.accuracy,
        metrics.f1,
        MODEL_PATH,
    )

    return ModelTrainingInfo(
        accuracy=metrics.accuracy,
        f1=metrics.f1,
    )


@app.get("/model/status", response_model=ModelStatus)
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


@app.get(
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
    history = load_training_history(TRAINING_HISTORY_PATH)
    if model_type is not None:
        history = [
            entry for entry in history if entry.model_type == model_type
        ]

    newest_first = list(reversed(history))
    return ModelMetricsResponse(
        latest=newest_first[0] if newest_first else None,
        history=newest_first[:limit],
    )
