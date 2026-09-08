from pathlib import Path

import httpx2
import pandas as pd
import pytest

from src import main
from src.dataset import ChurnDataset


pytestmark = pytest.mark.anyio


def make_synthetic_churn_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    for index in range(10):
        records.extend(
            [
                {
                    "monthly_fee": 20.0 + index,
                    "usage_hours": 40.0 + index,
                    "support_requests": index % 2,
                    "account_age_months": 24 + index,
                    "failed_payments": 0,
                    "region": "europe",
                    "device_type": "desktop",
                    "payment_method": "card",
                    "autopay_enabled": 1,
                    "churn": 0,
                },
                {
                    "monthly_fee": 80.0 + index,
                    "usage_hours": 5.0 + index,
                    "support_requests": 4 + index % 3,
                    "account_age_months": 1 + index,
                    "failed_payments": 2 + index % 2,
                    "region": "america",
                    "device_type": "mobile",
                    "payment_method": "paypal",
                    "autopay_enabled": 0,
                    "churn": 1,
                },
            ]
        )

    return records


async def test_day12_complete_http_churn_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "churn_dataset.csv"
    model_path = tmp_path / "models" / "churn_model.joblib"
    history_path = tmp_path / "models" / "training_history.json"
    records = make_synthetic_churn_records()
    pd.DataFrame.from_records(records).to_csv(dataset_path, index=False)

    monkeypatch.setattr(main, "DATASET_PATH", dataset_path)
    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    monkeypatch.setattr(main, "TRAINING_HISTORY_PATH", history_path)

    missing_state = object()
    original_dataset = getattr(main.app.state, "churn_dataset", missing_state)
    original_model = getattr(main.app.state, "churn_model", missing_state)
    original_overrides = main.app.dependency_overrides.copy()
    main.app.dependency_overrides.clear()
    transport = httpx2.ASGITransport(app=main.app)

    prediction_payload = {
        name: value for name, value in records[0].items() if name != "churn"
    }

    try:
        async with main.lifespan(main.app):
            loaded_dataset = main.app.state.churn_dataset
            assert isinstance(loaded_dataset, ChurnDataset)
            assert loaded_dataset.path == dataset_path
            assert len(loaded_dataset.dataframe) == len(records)

            async with httpx2.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                unavailable_response = await client.post(
                    "/predict",
                    json=prediction_payload,
                )
                assert unavailable_response.status_code == 503
                assert unavailable_response.json()["code"] == "model_not_trained"

                training_response = await client.post(
                    "/model/train",
                    json={"model_type": "logreg", "hyperparameters": {}},
                )
                assert training_response.status_code == 200
                training_metrics = training_response.json()
                assert 0 <= training_metrics["accuracy"] <= 1
                assert 0 <= training_metrics["f1"] <= 1

                status_response = await client.get("/model/status")
                assert status_response.status_code == 200
                status_payload = status_response.json()
                assert status_payload["is_trained"] is True
                assert status_payload["model_type"] == "logreg"
                assert status_payload["metrics"] == training_metrics

                prediction_response = await client.post(
                    "/predict",
                    json=prediction_payload,
                )
                assert prediction_response.status_code == 200
                prediction = prediction_response.json()
                assert prediction["predicted_class"] in {0, 1}
                assert set(prediction["class_probabilities"]) == {"0", "1"}
                assert sum(prediction["class_probabilities"].values()) == (
                    pytest.approx(1.0)
                )

            assert model_path.is_file()
            assert history_path.is_file()
            assert model_path.is_relative_to(tmp_path)
            assert history_path.is_relative_to(tmp_path)
    finally:
        main.app.dependency_overrides.clear()
        main.app.dependency_overrides.update(original_overrides)
        if original_dataset is missing_state:
            del main.app.state.churn_dataset
        else:
            main.app.state.churn_dataset = original_dataset
        if original_model is missing_state:
            del main.app.state.churn_model
        else:
            main.app.state.churn_model = original_model
