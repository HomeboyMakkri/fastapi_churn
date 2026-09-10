"""API routers for the churn service."""

from .dataset import router as dataset_router
from .model import router as model_router
from .prediction import router as prediction_router
from .system import router as system_router


__all__ = [
    "dataset_router",
    "model_router",
    "prediction_router",
    "system_router",
]
