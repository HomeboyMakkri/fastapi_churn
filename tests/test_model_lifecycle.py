from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import Request

from src import main
from src.dataset import ChurnDataset
from src.model_store import ChurnModelArtifact, load_churn_model


@pytest.mark.anyio
async def test_training_persists_model_and_next_lifespan_restores_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "models" / "churn_model.joblib"
    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    dataset = ChurnDataset(main.DATASET_PATH)
    dataset.load()
    request = cast(Request, SimpleNamespace(app=main.app))

    metrics = main.train_model(request=request, dataset=dataset)

    in_memory_artifact = main.app.state.churn_model
    persisted_artifact = load_churn_model(model_path)
    sample = dataset.dataframe.drop(columns="churn").iloc[[0]]
    assert isinstance(in_memory_artifact, ChurnModelArtifact)
    assert in_memory_artifact.trained_at.utcoffset() is not None
    assert persisted_artifact.accuracy == metrics.accuracy
    assert persisted_artifact.f1 == metrics.f1

    monkeypatch.setattr(main, "DATASET_PATH", tmp_path / "missing.csv")
    main.app.state.churn_model = None
    async with main.lifespan(main.app):
        restored_artifact = main.app.state.churn_model

        assert main.app.state.churn_dataset is None
        assert isinstance(restored_artifact, ChurnModelArtifact)
        assert restored_artifact.trained_at == persisted_artifact.trained_at
        assert restored_artifact.pipeline.predict(sample).tolist() == (
            persisted_artifact.pipeline.predict(sample).tolist()
        )
