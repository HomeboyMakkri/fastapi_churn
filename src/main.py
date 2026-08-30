from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, cast

from fastapi import Depends, FastAPI, Query, Request

from .dataset import ChurnDataset
from .preprocessing import (
    get_class_distribution,
    get_class_percentage,
    prepare_and_split,
)
from .schemas import (
    DatasetInfo,
    DatasetRowChurn,
    DatasetSplitInfo,
    FeatureVectorChurn,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "churn_dataset.csv"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    dataset = ChurnDataset(DATASET_PATH)
    dataset.load()
    app.state.churn_dataset = dataset
    yield


app = FastAPI(
    title="ML Churn Server",
    description="A FastAPI server for churn prediction and dataset management",
    version="1.0.0",
    lifespan=lifespan,
)


def get_dataset(request: Request) -> ChurnDataset:
    return cast(ChurnDataset, request.app.state.churn_dataset)


DatasetDependency = Annotated[ChurnDataset, Depends(get_dataset)]
PreviewCount = Annotated[
    int,
    Query(ge=1, le=100, description="Number of rows to preview"),
]


@app.get("/")
def read_root():
    return {"message": "ml churn server is running"}


@app.post("/predict", response_model=FeatureVectorChurn)
def predict_churn(feature_vector: FeatureVectorChurn) -> FeatureVectorChurn:
    return feature_vector


@app.get("/dataset/preview", response_model=list[DatasetRowChurn])
def preview_dataset(
    dataset: DatasetDependency,
    count: PreviewCount = 5,
) -> list[DatasetRowChurn]:
    return dataset.to_rows()[:count]


@app.get("/dataset/info", response_model=DatasetInfo)
def get_dataset_info(dataset: DatasetDependency) -> DatasetInfo:
    dataframe = dataset.dataframe
    total_rows, total_columns = dataframe.shape
    target = dataframe["churn"]

    return DatasetInfo(
        total_rows=total_rows,
        total_columns=total_columns,
        column_names=dataframe.columns.tolist(),
        churn_distribution=get_class_distribution(target),
        churn_percentage=get_class_percentage(target),
    )


@app.get("/dataset/split-info", response_model=DatasetSplitInfo)
def get_dataset_split_info(
    dataset: DatasetDependency,
) -> DatasetSplitInfo:
    X_train, X_test, y_train, y_test = prepare_and_split(dataset.dataframe)

    return DatasetSplitInfo(
        train_rows=len(X_train),
        test_rows=len(X_test),
        feature_count=X_train.shape[1],
        train_churn_distribution=get_class_distribution(y_train),
        test_churn_distribution=get_class_distribution(y_test),
        train_churn_percentage=get_class_percentage(y_train),
        test_churn_percentage=get_class_percentage(y_test),
    )
