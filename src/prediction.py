"""Run churn inference for validated feature vectors."""

from collections.abc import Sequence
from typing import cast

import pandas as pd

from .model_store import ChurnModelArtifact
from .preprocessing import FEATURES
from .schemas import (
    ChurnClass,
    ChurnClassLabel,
    ClassProbabilities,
    FeatureVectorChurn,
    PredictionResponseChurn,
)


def predict_churn_batch(
    artifact: ChurnModelArtifact,
    feature_vectors: Sequence[FeatureVectorChurn],
) -> list[PredictionResponseChurn]:
    """Predict churn classes and class probabilities for one non-empty batch."""
    if not isinstance(artifact, ChurnModelArtifact):
        raise TypeError("artifact must be a ChurnModelArtifact")
    if not feature_vectors:
        raise ValueError("At least one feature vector is required")
    if any(not isinstance(item, FeatureVectorChurn) for item in feature_vectors):
        raise TypeError("feature_vectors must contain FeatureVectorChurn objects")

    records = [item.model_dump() for item in feature_vectors]
    dataframe = pd.DataFrame.from_records(records, columns=list(FEATURES))
    predictions = artifact.pipeline.predict(dataframe)
    probabilities = artifact.pipeline.predict_proba(dataframe)
    model_classes = list(artifact.pipeline.classes_)

    if set(model_classes) != {0, 1} or len(model_classes) != 2:
        raise ValueError("Churn model must expose exactly classes 0 and 1")
    class_labels: list[ChurnClassLabel] = [
        "0" if value == 0 else "1" for value in model_classes
    ]
    if probabilities.shape != (len(feature_vectors), len(class_labels)):
        raise ValueError("Churn model returned an invalid probability matrix")
    if len(predictions) != len(feature_vectors):
        raise ValueError("Churn model returned an invalid number of predictions")

    responses: list[PredictionResponseChurn] = []
    for predicted_class, row_probabilities in zip(
        predictions,
        probabilities,
        strict=True,
    ):
        predicted_value = int(predicted_class)
        if predicted_value not in (0, 1):
            raise ValueError("Churn model returned a class other than 0 or 1")

        class_probabilities: ClassProbabilities = {
            class_label: float(probability)
            for class_label, probability in zip(
                class_labels,
                row_probabilities,
                strict=True,
            )
        }
        responses.append(
            PredictionResponseChurn(
                predicted_class=cast(ChurnClass, predicted_value),
                class_probabilities=class_probabilities,
            )
        )

    return responses
