import pytest
from pydantic import TypeAdapter, ValidationError

from src import main
from src.model_store import ChurnModelArtifact
from src.schemas import (
    FeatureVectorChurn,
    PredictionPayload,
    PredictionResponseChurn,
)


def make_feature_vector(
    valid_record: dict[str, object],
) -> FeatureVectorChurn:
    return FeatureVectorChurn.model_validate(
        {key: value for key, value in valid_record.items() if key != "churn"}
    )


def test_predict_endpoint_preserves_single_input_shape(
    valid_record: dict[str, object],
    trained_artifact: ChurnModelArtifact,
) -> None:
    feature_vector = make_feature_vector(valid_record)

    response = main.predict_churn(feature_vector, trained_artifact)

    assert isinstance(response, PredictionResponseChurn)
    assert response.predicted_class in {0, 1}
    assert sum(response.class_probabilities.values()) == pytest.approx(1.0)


def test_predict_endpoint_preserves_batch_order(
    valid_record: dict[str, object],
    trained_artifact: ChurnModelArtifact,
) -> None:
    first = make_feature_vector(valid_record)
    second = first.model_copy(update={"monthly_fee": 99.99})

    batch_response = main.predict_churn([first, second], trained_artifact)
    first_response = main.predict_churn(first, trained_artifact)
    second_response = main.predict_churn(second, trained_artifact)

    assert isinstance(batch_response, list)
    assert batch_response == [first_response, second_response]


def test_prediction_payload_rejects_empty_batch() -> None:
    adapter = TypeAdapter(PredictionPayload)

    with pytest.raises(ValidationError, match="at least 1 item"):
        adapter.validate_python([])


def test_predict_openapi_contains_single_and_batch_examples() -> None:
    operation = main.app.openapi()["paths"]["/predict"]["post"]
    request_content = operation["requestBody"]["content"]["application/json"]
    response_content = operation["responses"]["200"]["content"]["application/json"]

    assert set(request_content["examples"]) == {
        "single_customer",
        "customer_batch",
    }
    assert set(response_content["examples"]) == {
        "single_customer",
        "customer_batch",
    }
    assert "503" in operation["responses"]
