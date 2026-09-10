"""Global exception handlers for the churn service API."""

import logging
from typing import cast

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .errors import ApiHTTPException
from .schemas import ErrorDetail, ErrorResponse


logger = logging.getLogger("uvicorn.error.src.main")


async def request_validation_exception_handler(
    request: Request,
    exception: RequestValidationError,
) -> JSONResponse:
    """Return Pydantic request errors using the service error contract."""
    details = [
        ErrorDetail(
            location=list(error["loc"]),
            message=error["msg"],
            error_type=error["type"],
        )
        for error in exception.errors()
    ]
    payload = ErrorResponse(
        code="request_validation_error",
        message="Request data is invalid",
        details=details,
    )
    logger.warning(
        "Request validation failed during %s %s: status=%d code=%s",
        request.method,
        request.url.path,
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        payload.code,
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=payload.model_dump(mode="json"),
    )


async def http_exception_handler(
    request: Request,
    exception: HTTPException,
) -> JSONResponse:
    """Normalize service and ordinary FastAPI HTTP exceptions."""
    details: list[ErrorDetail] | dict[str, object] | None
    if isinstance(exception, ApiHTTPException):
        code = exception.code
        message = exception.message
        details = exception.details
    else:
        code = f"http_{exception.status_code}"
        message = (
            exception.detail
            if isinstance(exception.detail, str)
            else "HTTP request failed"
        )
        details = (
            None
            if isinstance(exception.detail, str)
            else {"detail": cast(object, exception.detail)}
        )

    payload = ErrorResponse(code=code, message=message, details=details)
    logger.warning(
        "HTTP error during %s %s: status=%d code=%s",
        request.method,
        request.url.path,
        exception.status_code,
        payload.code,
    )
    return JSONResponse(
        status_code=exception.status_code,
        content=payload.model_dump(mode="json"),
        headers=exception.headers,
    )


async def unhandled_exception_handler(
    request: Request,
    exception: Exception,
) -> JSONResponse:
    """Hide implementation details while retaining the traceback in logs."""
    logger.error(
        "Unhandled error during %s %s",
        request.method,
        request.url.path,
        exc_info=(type(exception), exception, exception.__traceback__),
    )
    payload = ErrorResponse(
        code="internal_server_error",
        message="An unexpected server error occurred",
        details=None,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=payload.model_dump(mode="json"),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register the service's global exception handlers on an application."""
    app.add_exception_handler(
        RequestValidationError,
        request_validation_exception_handler,
    )
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
