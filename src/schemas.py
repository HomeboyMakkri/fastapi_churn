from pydantic import BaseModel, Field

class FeatureVectorChurn(BaseModel):
    monthly_fee : float = Field(..., description="Monthly fee of the customer")
    usage_hours : float = Field(..., description="Usage hours of the customer")
    support_requests : int = Field(..., description="Number of support requests made by the customer")
    account_age_months : int = Field(..., description="Age of the customer's account in months")
    failed_payments : int = Field(..., description="Number of failed payments made by the customer")
    region : str = Field(..., description="Region of the customer")
    device_type : str = Field(..., description="Type of device used by the customer")
    payment_method : str = Field(..., description="Payment method used by the customer")
    autopay_enabled : int = Field(..., description="Whether autopay is enabled for the customer")

class DatasetRowChurn(FeatureVectorChurn):
    churn: int = Field(..., description="Predicted probability of churn for the customer")