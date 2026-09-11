"""FastAPI dependencies for accessing churn service runtime state."""

from typing import Annotated, cast

from fastapi import Depends, Query, Request

from .config import AppSettings
from .dataset import ChurnDataset
from .errors import DatasetUnavailableError, ModelNotTrainedError
from .model_store import ChurnModelArtifact


def get_settings(request: Request) -> AppSettings:
    return cast(AppSettings, request.app.state.settings)


def get_dataset(request: Request) -> ChurnDataset:
    dataset = getattr(request.app.state, "churn_dataset", None)
    if dataset is None:
        raise DatasetUnavailableError

    return cast(ChurnDataset, dataset)


DatasetDependency = Annotated[ChurnDataset, Depends(get_dataset)]


def get_churn_model(request: Request) -> ChurnModelArtifact:
    artifact = getattr(request.app.state, "churn_model", None)
    if artifact is None:
        raise ModelNotTrainedError

    return cast(ChurnModelArtifact, artifact)


ModelDependency = Annotated[ChurnModelArtifact, Depends(get_churn_model)]
PreviewCount = Annotated[
    int,
    Query(ge=1, le=100, description="Number of rows to preview"),
]
