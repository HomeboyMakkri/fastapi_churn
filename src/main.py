from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import cast

from fastapi import FastAPI

from .dataset import ChurnDataset
from .dependencies import (
    get_churn_model,
    get_dataset,
)
from .evaluation import evaluate_churn_model
from .exception_handlers import (
    http_exception_handler,
    register_exception_handlers,
    request_validation_exception_handler,
    unhandled_exception_handler,
)
from .model import train_churn_model
from .model_store import (
    ChurnModelArtifact,
    ModelPersistenceError,
    load_churn_model,
    save_churn_model,
)
from .prediction import predict_churn_batch
from .preprocessing import (
    prepare_and_split,
)
from .routers import (
    dataset_router,
    model_router,
    prediction_router,
    system_router,
)
from .routers.dataset import (
    get_dataset_info,
    get_dataset_split_info,
    preview_dataset,
)
from .routers.model import (
    get_model_metrics,
    get_model_schema,
    get_model_status,
    train_model,
)
from .routers.prediction import predict_churn
from .routers.system import get_health, read_root
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
app.include_router(system_router)
app.include_router(dataset_router)
app.include_router(prediction_router)
app.include_router(model_router)
