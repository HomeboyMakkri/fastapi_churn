from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest
from fastapi import Request
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src import main
from src.dataset import ChurnDataset
from src.evaluation import evaluate_churn_model
from src.model import train_churn_model
from src.preprocessing import prepare_and_split
from src.schemas import TrainingConfigChurn, TrainingMetrics


def make_learnable_dataframe(row_count: int = 100) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for index in range(row_count):
        churn = index % 2
        records.append(
            {
                "monthly_fee": 30.0,
                "usage_hours": 5.0 if churn else 45.0,
                "support_requests": 2 if churn else 0,
                "account_age_months": 12,
                "failed_payments": churn,
                "region": "europe",
                "device_type": "mobile",
                "payment_method": "card",
                "autopay_enabled": 1,
                "churn": churn,
            }
        )
    return pd.DataFrame.from_records(records)


def test_evaluate_churn_model_matches_sklearn_metrics() -> None:
    X_train, X_test, y_train, y_test = prepare_and_split(
        make_learnable_dataframe()
    )
    pipeline = train_churn_model(X_train, y_train)

    metrics = evaluate_churn_model(pipeline, X_test, y_test)

    predictions = pipeline.predict(X_test)
    positive_class_index = list(pipeline.classes_).index(1)
    positive_probabilities = pipeline.predict_proba(X_test)[
        :, positive_class_index
    ]
    assert metrics == TrainingMetrics(
        accuracy=accuracy_score(y_test, predictions),
        f1=f1_score(y_test, predictions),
        roc_auc=roc_auc_score(y_test, positive_probabilities),
    )


def test_evaluate_churn_model_finds_positive_probability_column() -> None:
    class ReversedClassesPipeline:
        classes_ = np.array([1, 0])

        def predict(self, _features: pd.DataFrame) -> np.ndarray:
            return np.array([0, 0, 1, 1])

        def predict_proba(self, _features: pd.DataFrame) -> np.ndarray:
            return np.array(
                [
                    [0.1, 0.9],
                    [0.2, 0.8],
                    [0.8, 0.2],
                    [0.9, 0.1],
                ]
            )

    pipeline = cast(Pipeline, ReversedClassesPipeline())
    X_test = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y_test = pd.Series([0, 0, 1, 1])

    metrics = evaluate_churn_model(pipeline, X_test, y_test)

    assert metrics.roc_auc == 1.0


def test_evaluate_churn_model_rejects_model_without_churn_class() -> None:
    class SingleClassPipeline:
        classes_ = np.array([0])

        def predict(self, _features: pd.DataFrame) -> np.ndarray:
            return np.array([0, 0])

        def predict_proba(self, _features: pd.DataFrame) -> np.ndarray:
            return np.array([[1.0], [1.0]])

    with pytest.raises(ValueError, match="does not contain churn class 1"):
        evaluate_churn_model(
            cast(Pipeline, SingleClassPipeline()),
            pd.DataFrame({"feature": [1, 2]}),
            pd.Series([0, 1]),
        )


def test_train_model_evaluates_only_held_out_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = ChurnDataset(main.DATASET_PATH)
    dataset.load()
    expected_split = prepare_and_split(dataset.dataframe)
    expected_X_test = expected_split[1]
    expected_y_test = expected_split[3]
    evaluated: dict[str, pd.DataFrame | pd.Series] = {}
    real_evaluate = evaluate_churn_model

    def capture_evaluation(
        pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> TrainingMetrics:
        evaluated["features"] = X_test
        evaluated["target"] = y_test
        return real_evaluate(pipeline, X_test, y_test)

    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "churn_model.joblib")
    monkeypatch.setattr(
        main,
        "TRAINING_HISTORY_PATH",
        tmp_path / "training_history.json",
    )
    monkeypatch.setattr(main, "evaluate_churn_model", capture_evaluation)
    app_stub = SimpleNamespace(state=SimpleNamespace(churn_model=None))
    request = cast(Request, SimpleNamespace(app=app_stub))

    result = main.train_model(
        request=request,
        config=TrainingConfigChurn(model_type="logreg"),
        dataset=dataset,
    )

    pd.testing.assert_frame_equal(evaluated["features"], expected_X_test)
    pd.testing.assert_series_equal(evaluated["target"], expected_y_test)
    assert result.model_dump() == {
        "accuracy": pytest.approx(0.7875),
        "f1": pytest.approx(0.0449438202247191),
    }
