"""FastAPI application factory for the churn service."""

from fastapi import FastAPI

from .config import AppSettings
from .exception_handlers import register_exception_handlers
from .lifespan import lifespan
from .routers import (
    dataset_router,
    model_router,
    prediction_router,
    system_router,
)


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Create and configure an independent churn-service application."""
    app = FastAPI(
        title="ML Churn Server",
        description="A FastAPI server for churn prediction and dataset management",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings if settings is not None else AppSettings()
    register_exception_handlers(app)
    app.include_router(system_router)
    app.include_router(dataset_router)
    app.include_router(prediction_router)
    app.include_router(model_router)
    return app
