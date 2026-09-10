"""OpenAPI examples for the churn service."""

from fastapi.openapi.models import Example


PREDICTION_REQUEST_EXAMPLES: dict[str, Example] = {
    "single_customer": {
        "summary": "One customer",
        "value": {
            "monthly_fee": 79.99,
            "usage_hours": 8.5,
            "support_requests": 4,
            "account_age_months": 6,
            "failed_payments": 2,
            "region": "europe",
            "device_type": "mobile",
            "payment_method": "card",
            "autopay_enabled": 0,
        },
    },
    "customer_batch": {
        "summary": "Several customers",
        "value": [
            {
                "monthly_fee": 29.99,
                "usage_hours": 45.0,
                "support_requests": 0,
                "account_age_months": 36,
                "failed_payments": 0,
                "region": "europe",
                "device_type": "desktop",
                "payment_method": "card",
                "autopay_enabled": 1,
            },
            {
                "monthly_fee": 99.99,
                "usage_hours": 4.0,
                "support_requests": 6,
                "account_age_months": 2,
                "failed_payments": 3,
                "region": "america",
                "device_type": "mobile",
                "payment_method": "paypal",
                "autopay_enabled": 0,
            },
        ],
    },
}

PREDICTION_RESPONSE_EXAMPLES = {
    "single_customer": {
        "summary": "Prediction for one customer",
        "value": {
            "predicted_class": 1,
            "class_probabilities": {"0": 0.23, "1": 0.77},
        },
    },
    "customer_batch": {
        "summary": "Predictions in request order",
        "value": [
            {
                "predicted_class": 0,
                "class_probabilities": {"0": 0.84, "1": 0.16},
            },
            {
                "predicted_class": 1,
                "class_probabilities": {"0": 0.31, "1": 0.69},
            },
        ],
    },
}

PREDICTION_VALIDATION_ERROR_EXAMPLE = {
    "code": "request_validation_error",
    "message": "Request data is invalid",
    "details": [
        {
            "location": ["body", "monthly_fee"],
            "message": "Field required",
            "error_type": "missing",
        }
    ],
}
MODEL_NOT_TRAINED_ERROR_EXAMPLE = {
    "code": "model_not_trained",
    "message": "Churn model is not trained",
    "details": None,
}
PREDICTION_FAILED_ERROR_EXAMPLE = {
    "code": "prediction_failed",
    "message": "Could not calculate churn prediction",
    "details": None,
}
MODEL_CONFIGURATION_ERROR_EXAMPLE = {
    "code": "model_configuration_error",
    "message": "Model configuration is invalid",
    "details": {"reason": "Unsupported hyperparameters for logreg: unknown"},
}
DATASET_EMPTY_ERROR_EXAMPLE = {
    "code": "dataset_empty",
    "message": "Churn dataset is empty",
    "details": None,
}
INTERNAL_SERVER_ERROR_EXAMPLE = {
    "code": "internal_server_error",
    "message": "An unexpected server error occurred",
    "details": None,
}
