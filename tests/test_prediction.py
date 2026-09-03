from datetime import datetime, timezone
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from src.model_store import ChurnModelArtifact
from src.prediction import predict_churn_batch
from src.preprocessing import FEATURES
from src.schemas import FeatureVectorChurn


class ReversedClassClassifier(ClassifierMixin, BaseEstimator):
    """Deterministic test classifier whose probability columns are [1, 0]."""

    def fit(self, X: pd.DataFrame, y: list[int]) -> "ReversedClassClassifier":
        self.classes_ = np.array([1, 0])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.where(X["monthly_fee"].to_numpy() >= 50, 1, 0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        high_fee = X["monthly_fee"].to_numpy() >= 50
        class_one = np.where(high_fee, 0.8, 0.1)
        class_zero = 1.0 - class_one
        return np.column_stack([class_one, class_zero])


def make_artifact() -> ChurnModelArtifact:
    training_frame = pd.DataFrame(
        [
            {feature: 0 for feature in FEATURES},
            {feature: 1 for feature in FEATURES},
        ]
    )
    pipeline = Pipeline([("classifier", ReversedClassClassifier())])
    pipeline.fit(training_frame, [0, 1])
    return ChurnModelArtifact(
        pipeline=pipeline,
        trained_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
        accuracy=0.8,
        f1=0.7,
        model_type="logreg",
        hyperparameters={},
    )


def make_feature_vector(
    valid_record: dict[str, object],
    monthly_fee: float,
) -> FeatureVectorChurn:
    record = {
        key: value
        for key, value in valid_record.items()
        if key != "churn"
    }
    record["monthly_fee"] = monthly_fee
    return FeatureVectorChurn.model_validate(record)


def test_predict_churn_batch_preserves_order_and_maps_model_classes(
    valid_record: dict[str, object],
) -> None:
    low_fee = make_feature_vector(valid_record, monthly_fee=10.0)
    high_fee = make_feature_vector(valid_record, monthly_fee=90.0)

    responses = predict_churn_batch(make_artifact(), [low_fee, high_fee])

    assert [response.predicted_class for response in responses] == [0, 1]
    assert responses[0].class_probabilities == pytest.approx(
        {"1": 0.1, "0": 0.9}
    )
    assert responses[1].class_probabilities == pytest.approx(
        {"1": 0.8, "0": 0.2}
    )


def test_predict_churn_batch_passes_features_in_contract_order(
    valid_record: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = make_artifact()
    feature_vector = make_feature_vector(valid_record, monthly_fee=10.0)
    observed_columns: list[list[str]] = []
    original_predict = artifact.pipeline.predict

    def capture_predict(dataframe: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        observed_columns.append(dataframe.columns.tolist())
        return cast(np.ndarray, original_predict(dataframe, **kwargs))

    monkeypatch.setattr(artifact.pipeline, "predict", capture_predict)

    predict_churn_batch(artifact, [feature_vector])

    assert observed_columns == [list(FEATURES)]


def test_predict_churn_batch_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="At least one feature vector"):
        predict_churn_batch(make_artifact(), [])


def test_predict_churn_batch_rejects_non_feature_vector() -> None:
    with pytest.raises(TypeError, match="FeatureVectorChurn"):
        predict_churn_batch(make_artifact(), [{}])  # type: ignore[list-item]
