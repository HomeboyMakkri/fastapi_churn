from datetime import datetime
from math import isclose
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
ChurnClass = Literal[0, 1]
ChurnClassLabel = Literal["0", "1"]
ClassProbabilities = dict[ChurnClassLabel, Probability]
ModelType = Literal["logreg", "random_forest"]
HyperparameterValue = str | int | FiniteFloat | bool | None
Hyperparameters = dict[str, HyperparameterValue]


class FeatureVectorChurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    monthly_fee: float = Field(
        ..., ge=0, allow_inf_nan=False, description="Monthly fee of the customer"
    )
    usage_hours: float = Field(
        ..., ge=0, allow_inf_nan=False, description="Usage hours of the customer"
    )
    support_requests: int = Field(
        ..., ge=0, description="Number of support requests made by the customer"
    )
    account_age_months: int = Field(
        ..., ge=0, description="Age of the customer's account in months"
    )
    failed_payments: int = Field(
        ..., ge=0, description="Number of failed payments made by the customer"
    )
    region: Literal["europe", "asia", "america", "africa"] = Field(
        ..., description="Region of the customer"
    )
    device_type: Literal["mobile", "desktop", "tablet"] = Field(
        ..., description="Type of device used by the customer"
    )
    payment_method: Literal["card", "paypal", "crypto"] = Field(
        ..., description="Payment method used by the customer"
    )
    autopay_enabled: Literal[0, 1] = Field(
        ..., description="Whether autopay is enabled for the customer"
    )


FeatureVectorBatch = Annotated[list[FeatureVectorChurn], Field(min_length=1)]
PredictionPayload = FeatureVectorChurn | FeatureVectorBatch


class PredictionResponseChurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_class: ChurnClass = Field(
        ..., description="Predicted churn class: 0 or 1"
    )
    class_probabilities: ClassProbabilities = Field(
        ..., description="Probability assigned to each churn class"
    )

    @model_validator(mode="after")
    def validate_class_probabilities(self) -> Self:
        if set(self.class_probabilities) != {"0", "1"}:
            raise ValueError("Probabilities must be provided for classes 0 and 1")
        if not isclose(
            sum(self.class_probabilities.values()),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("Class probabilities must sum to 1")
        return self


PredictionResult = PredictionResponseChurn | list[PredictionResponseChurn]


class DatasetRowChurn(FeatureVectorChurn):
    churn: Literal[0, 1] = Field(
        ..., description="Whether the customer churned: 0 or 1"
    )


class DatasetInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_rows: int = Field(
        ..., ge=0, description="Total number of rows in the dataset"
    )
    total_columns: int = Field(
        ..., ge=0, description="Total number of columns in the dataset"
    )
    column_names: list[str] = Field(
        ..., description="List of column names in the dataset"
    )
    churn_distribution: dict[str, int] = Field(
        ...,
        description="Distribution of churn values in the dataset"
    )
    churn_percentage: dict[str, float] = Field(
        ...,
        description="Percentage of churn values in the dataset"
    )


class DatasetSplitInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_rows: int = Field(..., ge=1, description="Number of training rows")
    test_rows: int = Field(..., ge=1, description="Number of test rows")
    feature_count: int = Field(..., ge=1, description="Number of input features")
    train_churn_distribution: dict[str, int] = Field(
        ..., description="Churn class counts in the training set"
    )
    test_churn_distribution: dict[str, int] = Field(
        ..., description="Churn class counts in the test set"
    )
    train_churn_percentage: dict[str, float] = Field(
        ..., description="Churn class percentages in the training set"
    )
    test_churn_percentage: dict[str, float] = Field(
        ..., description="Churn class percentages in the test set"
    )


class TrainingConfigChurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_type: ModelType = Field(
        ..., description="Type of churn classifier to train"
    )
    hyperparameters: Hyperparameters = Field(
        default_factory=dict,
        description="Hyperparameters passed to the churn classifier",
    )


class ModelTrainingInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accuracy: float = Field(
        ..., ge=0, le=1, description="Accuracy of the trained model"
    )
    f1: float = Field(
        ..., ge=0, le=1, description="F1 score of the trained model"
    )


class ModelStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_trained: bool = Field(
        ..., description="Whether a trained churn model is available"
    )
    last_trained_at: datetime | None = Field(
        ..., description="UTC timestamp of the latest successful training"
    )
    metrics: ModelTrainingInfo | None = Field(
        ..., description="Metrics calculated on the test dataset"
    )

    @model_validator(mode="after")
    def validate_model_state(self) -> Self:
        has_training_time = self.last_trained_at is not None
        has_metrics = self.metrics is not None
        state_is_consistent = (
            self.is_trained and has_training_time and has_metrics
        ) or (
            not self.is_trained and not has_training_time and not has_metrics
        )
        if not state_is_consistent:
            raise ValueError(
                "Model training time and metrics must match its trained state"
            )
        return self
