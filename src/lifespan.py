"""Application lifespan for loading churn runtime resources."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging
from typing import cast

from fastapi import FastAPI

from .dataset import ChurnDataset
from .model_store import (
    ChurnModelArtifact,
    ModelPersistenceError,
    load_churn_model,
)


logger = logging.getLogger("uvicorn.error.src.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Reset runtime state, then independently load the dataset and model."""
    settings = app.state.settings
    dataset_path = settings.dataset_path
    model_path = settings.model_path

    app.state.churn_dataset = None
    app.state.churn_model = None

    dataset = ChurnDataset(dataset_path)
    logger.info("Loading churn dataset from %s", dataset_path)
    try:
        dataset.load()
    except (OSError, ValueError) as error:
        logger.warning(
            "Could not load churn dataset from %s: error_type=%s",
            dataset_path,
            type(error).__name__,
        )
    else:
        app.state.churn_dataset = dataset
        logger.info(
            "Loaded churn dataset from %s with %d rows",
            dataset_path,
            len(dataset.dataframe),
        )

    logger.info("Restoring churn model from %s", model_path)
    try:
        app.state.churn_model = load_churn_model(model_path)
    except FileNotFoundError:
        logger.info("No saved churn model found at %s", model_path)
    except (OSError, ValueError, ModelPersistenceError) as error:
        logger.warning(
            "Could not restore churn model from %s: error_type=%s",
            model_path,
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
