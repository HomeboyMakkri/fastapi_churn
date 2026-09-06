from datetime import datetime
from math import isclose
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    field_validator,
    model_validator,
)


Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
ChurnClass = Literal[0, 1]
ChurnClassLabel = Literal["0", "1"]
ClassProbabilities = dict[ChurnClassLabel, Probability]
ModelType = Literal["logreg", "random_forest"]
HyperparameterValue = str | int | FiniteFloat | bool | None
Hyperparameters = dict[str, HyperparameterValue]
FeatureValueType = Literal["number", "integer", "string"]
FeatureGroup = Literal["numeric", "categorical"]


class ErrorDetail(BaseModel):
    """Describe one concrete problem in an API request."""

    model_config = ConfigDict(extra="forbid")

    location: list[str | int] | None = Field(
        default=None,
        description="Location of the invalid value in the request",
    )
    message: str = Field(..., min_length=1, description="Problem description")
    error_type: str | None = Field(
        default=None,
        description="Stable validation-error type",
    )


class ErrorResponse(BaseModel):
    """Common response returned for every handled API error."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., min_length=1, description="Stable machine-readable code")
    message: str = Field(..., min_length=1, description="Human-readable message")
    details: list[ErrorDetail] | dict[str, object] | None = Field(
        default=None,
        description="Optional structured context about the error",
    )


class FeatureVectorChurn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

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


class ModelFeatureSchemaChurn(BaseModel):
    """Describe one input feature expected by the churn model."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Input feature name")
    value_type: FeatureValueType = Field(
        ..., description="JSON value type expected for the feature"
    )
    group: FeatureGroup = Field(
        ..., description="Preprocessing group assigned to the feature"
    )


class ModelSchemaChurn(BaseModel):
    """Describe the ordered input contract of the churn model."""

    model_config = ConfigDict(extra="forbid")

    features: list[ModelFeatureSchemaChurn] = Field(
        ...,
        min_length=1,
        description="Input features in the order used by the model",
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
        ...,
        ge=0,
        le=1,
        allow_inf_nan=False,
        description="Accuracy of the trained model",
    )
    f1: float = Field(
        ...,
        ge=0,
        le=1,
        allow_inf_nan=False,
        description="F1 score of the trained model",
    )


class TrainingMetrics(ModelTrainingInfo):
    """Quality metrics calculated for one completed training run."""

    roc_auc: float = Field(
        ...,
        ge=0,
        le=1,
        allow_inf_nan=False,
        description="ROC AUC of the trained model",
    )


class TrainingHistoryEntry(BaseModel):
    """Serializable metadata and metrics for one completed training run."""

    model_config = ConfigDict(extra="forbid")

    trained_at: datetime = Field(
        ..., description="Timezone-aware timestamp of the training run"
    )
    model_type: ModelType = Field(
        ..., description="Type of churn classifier that was trained"
    )
    hyperparameters: Hyperparameters = Field(
        ..., description="Hyperparameters requested for the classifier"
    )
    metrics: TrainingMetrics = Field(
        ..., description="Metrics calculated on the test dataset"
    )

    @field_validator("trained_at")
    @classmethod
    def validate_trained_at_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("trained_at must include timezone information")
        return value


class ModelMetricsResponse(BaseModel):
    """Latest training result and a bounded view of training history."""

    model_config = ConfigDict(extra="forbid")

    latest: TrainingHistoryEntry | None = Field(
        ..., description="Latest training record, if one is available"
    )
    history: list[TrainingHistoryEntry] = Field(
        ..., description="Training records ordered from newest to oldest"
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
    model_type: ModelType | None = Field(
        ..., description="Type of the trained churn classifier"
    )
    hyperparameters: Hyperparameters | None = Field(
        ..., description="Hyperparameters requested for the trained classifier"
    )

    @model_validator(mode="after")
    def validate_model_state(self) -> Self:
        metadata = (
            self.last_trained_at,
            self.metrics,
            self.model_type,
            self.hyperparameters,
        )
        has_all_metadata = all(value is not None for value in metadata)
        has_no_metadata = all(value is None for value in metadata)
        state_is_consistent = (
            self.is_trained and has_all_metadata
        ) or (not self.is_trained and has_no_metadata)
        if not state_is_consistent:
            raise ValueError(
                "Model metadata must match its trained state"
            )
        return self
