"""Churn prediction endpoint."""

from collections.abc import Callable
from typing import Annotated
import logging

from fastapi import APIRouter, Body

from ..dependencies import ModelDependency
from ..errors import PredictionError
from ..model_store import ChurnModelArtifact
from ..openapi_examples import (
    MODEL_NOT_TRAINED_ERROR_EXAMPLE,
    PREDICTION_FAILED_ERROR_EXAMPLE,
    PREDICTION_REQUEST_EXAMPLES,
    PREDICTION_RESPONSE_EXAMPLES,
    PREDICTION_VALIDATION_ERROR_EXAMPLE,
)
from ..prediction import predict_churn_batch
from ..schemas import (
    ErrorResponse,
    FeatureVectorChurn,
    PredictionPayload,
    PredictionResponseChurn,
    PredictionResult,
)


logger = logging.getLogger("uvicorn.error.src.main")
router = APIRouter()

PredictionRequest = Annotated[
    PredictionPayload,
    Body(openapi_examples=PREDICTION_REQUEST_EXAMPLES),
]
PredictionBatchFunction = Callable[
    [ChurnModelArtifact, list[FeatureVectorChurn]],
    list[PredictionResponseChurn],
]
_DEFAULT_PREDICTION_FUNCTION = predict_churn_batch


def _predict_churn(
    payload: PredictionPayload,
    artifact: ChurnModelArtifact,
    *,
    prediction_function: PredictionBatchFunction,
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
        predictions = prediction_function(artifact, feature_vectors)
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


@router.post(
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
    from ..main import predict_churn_batch as legacy_prediction_function

    prediction_function = (
        legacy_prediction_function
        if legacy_prediction_function is not _DEFAULT_PREDICTION_FUNCTION
        else predict_churn_batch
    )

    return _predict_churn(
        payload,
        artifact,
        prediction_function=prediction_function,
    )
