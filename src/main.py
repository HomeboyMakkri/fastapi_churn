from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, cast

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.openapi.models import Example
from sklearn.metrics import accuracy_score, f1_score

from .dataset import ChurnDataset
from .model import ModelConfigurationError, train_churn_model
from .model_store import (
    ChurnModelArtifact,
    ModelPersistenceError,
    load_churn_model,
    save_churn_model,
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
    FeatureVectorChurn,
    ModelStatus,
    ModelTrainingInfo,
    PredictionPayload,
    PredictionResponseChurn,
    PredictionResult,
    TrainingConfigChurn,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "churn_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"

PREDICTION_REQUEST_EXAMPLES: dict[str, Example] = {
    "single_customer": {
        "summary": "One customer",
        "value": {
            "monthly_fee": 79.99,
            "usage_hours": 8.5,
            "support_requests": 4,
            "account_age_months": 6,
            "failed_payments": 2,
            "region": "europe",
            "device_type": "mobile",
            "payment_method": "card",
            "autopay_enabled": 0,
        },
    },
    "customer_batch": {
        "summary": "Several customers",
        "value": [
            {
                "monthly_fee": 29.99,
                "usage_hours": 45.0,
                "support_requests": 0,
                "account_age_months": 36,
                "failed_payments": 0,
                "region": "europe",
                "device_type": "desktop",
                "payment_method": "card",
                "autopay_enabled": 1,
            },
            {
                "monthly_fee": 99.99,
                "usage_hours": 4.0,
                "support_requests": 6,
                "account_age_months": 2,
                "failed_payments": 3,
                "region": "america",
                "device_type": "mobile",
                "payment_method": "paypal",
                "autopay_enabled": 0,
            },
        ],
    },
}

PREDICTION_RESPONSE_EXAMPLES = {
    "single_customer": {
        "summary": "Prediction for one customer",
        "value": {
            "predicted_class": 1,
            "class_probabilities": {"0": 0.23, "1": 0.77},
        },
    },
    "customer_batch": {
        "summary": "Predictions in request order",
        "value": [
            {
                "predicted_class": 0,
                "class_probabilities": {"0": 0.84, "1": 0.16},
            },
            {
                "predicted_class": 1,
                "class_probabilities": {"0": 0.31, "1": 0.69},
            },
        ],
    },
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.churn_dataset = None
    app.state.churn_model = None

    dataset = ChurnDataset(DATASET_PATH)
    try:
        dataset.load()
    except (OSError, ValueError):
        pass
    else:
        app.state.churn_dataset = dataset

    try:
        app.state.churn_model = load_churn_model(MODEL_PATH)
    except (OSError, ValueError, ModelPersistenceError):
        pass

    yield


app = FastAPI(
    title="ML Churn Server",
    description="A FastAPI server for churn prediction and dataset management",
    version="1.0.0",
    lifespan=lifespan,
)


def get_dataset(request: Request) -> ChurnDataset:
    dataset = getattr(request.app.state, "churn_dataset", None)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Churn dataset is not available",
        )

    return cast(ChurnDataset, dataset)


DatasetDependency = Annotated[ChurnDataset, Depends(get_dataset)]


def get_churn_model(request: Request) -> ChurnModelArtifact:
    artifact = getattr(request.app.state, "churn_model", None)
    if artifact is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Churn model is not trained",
        )

    return cast(ChurnModelArtifact, artifact)


ModelDependency = Annotated[ChurnModelArtifact, Depends(get_churn_model)]
PreviewCount = Annotated[
    int,
    Query(ge=1, le=100, description="Number of rows to preview"),
]


@app.get("/")
def read_root():
    return {"message": "ml churn server is running"}


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
            "description": "A trained churn model is not available",
        },
    },
)
def predict_churn(
    payload: PredictionRequest,
    artifact: ModelDependency,
) -> PredictionResult:
    is_single_customer = isinstance(payload, FeatureVectorChurn)
    feature_vectors = [payload] if is_single_customer else payload
    predictions = predict_churn_batch(artifact, feature_vectors)

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


@app.post("/model/train", response_model=ModelTrainingInfo)
def train_model(
    request: Request,
    config: TrainingConfigChurn,
    dataset: DatasetDependency,
) -> ModelTrainingInfo:
    try:
        dataframe = dataset.dataframe
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Churn dataset is not loaded",
        ) from error

    if dataframe.empty:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Churn dataset is empty",
        )

    X_train, X_test, y_train, y_test = prepare_and_split(dataframe)
    try:
        pipeline = train_churn_model(
            X_train,
            y_train,
            model_type=config.model_type,
            hyperparameters=config.hyperparameters,
        )
    except ModelConfigurationError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error

    predictions = pipeline.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    f1 = float(f1_score(y_test, predictions))

    artifact = ChurnModelArtifact(
        pipeline=pipeline,
        trained_at=datetime.now(timezone.utc),
        accuracy=accuracy,
        f1=f1,
        model_type=config.model_type,
        hyperparameters=config.hyperparameters,
    )

    save_churn_model(artifact, MODEL_PATH)
    request.app.state.churn_model = artifact

    return ModelTrainingInfo(
        accuracy=accuracy,
        f1=f1,
    )


@app.get("/model/status", response_model=ModelStatus)
def get_model_status(request: Request) -> ModelStatus:
    artifact = getattr(request.app.state, "churn_model", None)
    if artifact is None:
        return ModelStatus(
            is_trained=False,
            last_trained_at=None,
            metrics=None,
        )

    artifact = cast(ChurnModelArtifact, artifact)
    return ModelStatus(
        is_trained=True,
        last_trained_at=artifact.trained_at,
        metrics=ModelTrainingInfo(
            accuracy=artifact.accuracy,
            f1=artifact.f1,
        ),
    )
