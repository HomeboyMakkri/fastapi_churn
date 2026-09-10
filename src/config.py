"""Application configuration for churn service instances."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Filesystem locations used by one application instance."""

    dataset_path: Path = PROJECT_ROOT / "data" / "churn_dataset.csv"
    model_path: Path = PROJECT_ROOT / "models" / "churn_model.joblib"
    training_history_path: Path = PROJECT_ROOT / "models" / "training_history.json"
