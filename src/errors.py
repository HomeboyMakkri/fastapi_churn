"""Public API exceptions for the churn service.

Low-level dataset and model functions deliberately keep raising ordinary
Python exceptions. Endpoints translate those exceptions into the classes in
this module, which form the service's public HTTP error contract.
"""

from collections.abc import Mapping

from fastapi import HTTPException, status

from .schemas import ErrorDetail


ErrorDetails = list[ErrorDetail] | dict[str, object] | None


class ApiHTTPException(HTTPException):
    """Base HTTP exception carrying the common public error fields."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: ErrorDetails = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            status_code=status_code,
            detail=message,
            headers=dict(headers) if headers is not None else None,
        )
        self.code = code
        self.message = message
        self.details = details


class DatasetUnavailableError(ApiHTTPException):
    def __init__(self, message: str = "Churn dataset is not available") -> None:
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="dataset_unavailable",
            message=message,
        )


class DatasetEmptyError(ApiHTTPException):
    def __init__(self) -> None:
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="dataset_empty",
            message="Churn dataset is empty",
        )


class ModelNotTrainedError(ApiHTTPException):
    def __init__(self) -> None:
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="model_not_trained",
            message="Churn model is not trained",
        )


class DataPreparationError(ApiHTTPException):
    def __init__(self, reason: str) -> None:
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="data_preparation_error",
            message="Training dataset cannot be prepared",
            details={"reason": reason},
        )


class ModelConfigurationApiError(ApiHTTPException):
    def __init__(self, reason: str) -> None:
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="model_configuration_error",
            message="Model configuration is invalid",
            details={"reason": reason},
        )


class PredictionError(ApiHTTPException):
    def __init__(self) -> None:
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="prediction_failed",
            message="Could not calculate churn prediction",
        )
