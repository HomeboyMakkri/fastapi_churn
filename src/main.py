"""ASGI entry point for the churn service."""

from .application import create_app


app = create_app()
