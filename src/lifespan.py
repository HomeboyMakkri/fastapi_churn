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
_DEFAULT_DATASET_CLASS = ChurnDataset
_DEFAULT_MODEL_LOADER = load_churn_model


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Reset runtime state, then independently load the dataset and model."""
    from . import main as main_module

    dataset_path = main_module.DATASET_PATH
    model_path = main_module.MODEL_PATH
    dataset_class = (
        main_module.ChurnDataset
        if main_module.ChurnDataset is not _DEFAULT_DATASET_CLASS
        else ChurnDataset
    )
    model_loader = (
        main_module.load_churn_model
        if main_module.load_churn_model is not _DEFAULT_MODEL_LOADER
        else load_churn_model
    )

    app.state.churn_dataset = None
    app.state.churn_model = None

    dataset = dataset_class(dataset_path)
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
        app.state.churn_model = model_loader(model_path)
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
