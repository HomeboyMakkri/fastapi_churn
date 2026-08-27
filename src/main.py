from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, cast

from fastapi import Depends, FastAPI, Query, Request

from .dataset import ChurnDataset
from .schemas import DatasetInfo, DatasetRowChurn, FeatureVectorChurn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "churn_dataset.csv"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
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
    churn_counts = dataframe["churn"].value_counts()

    churn_distribution = {
        str(churn_class): int(churn_counts.get(churn_class, 0))
        for churn_class in (0, 1)
    }
    churn_percentage = {
        churn_class: round((count / total_rows) * 100, 2)
        for churn_class, count in churn_distribution.items()
    }

    return DatasetInfo(
        total_rows=total_rows,
        total_columns=total_columns,
        column_names=dataframe.columns.tolist(),
        churn_distribution=churn_distribution,
        churn_percentage=churn_percentage,
    )
