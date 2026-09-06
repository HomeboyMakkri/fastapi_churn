"""Evaluate trained churn classifiers on held-out data."""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from .schemas import TrainingMetrics


def evaluate_churn_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> TrainingMetrics:
    """Calculate classification metrics using only the held-out test split."""
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)

    classes = list(pipeline.classes_)
    try:
        positive_class_index = classes.index(1)
    except ValueError as error:
        raise ValueError("Trained model does not contain churn class 1") from error

    positive_class_probabilities = probabilities[:, positive_class_index]
    return TrainingMetrics(
        accuracy=float(accuracy_score(y_test, predictions)),
        f1=float(f1_score(y_test, predictions)),
        roc_auc=float(roc_auc_score(y_test, positive_class_probabilities)),
    )
