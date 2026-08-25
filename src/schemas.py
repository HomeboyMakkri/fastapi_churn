from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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


class DatasetRowChurn(FeatureVectorChurn):
    churn: Literal[0, 1] = Field(
        ..., description="Whether the customer churned: 0 or 1"
    )
