import logging
from pathlib import Path

from .application import create_app
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
from .lifespan import lifespan
from .model import train_churn_model
from .model_store import (
    load_churn_model,
    save_churn_model,
)
from .prediction import predict_churn_batch
from .preprocessing import (
    prepare_and_split,
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


app = create_app()
