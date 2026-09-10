"""System endpoints for service status and discovery."""

from fastapi import APIRouter, Request

from ..schemas import HealthStatus


router = APIRouter()


@router.get("/")
def read_root():
    return {"message": "ml churn server is running"}


@router.get(
    "/health",
    response_model=HealthStatus,
    summary="Get churn service health",
    description=(
        "Returns whether the churn dataset and trained model are currently "
        "available. The endpoint remains available when either resource is missing."
    ),
)
def get_health(request: Request) -> HealthStatus:
    """Return the availability of runtime resources without requiring them."""
    return HealthStatus(
        model_available=getattr(request.app.state, "churn_model", None) is not None,
        dataset_loaded=getattr(request.app.state, "churn_dataset", None) is not None,
    )
