"""Shared FastAPI dependencies.

These are deliberately zero-argument callables. A dependency whose signature
takes a `Store` makes FastAPI try to bind that parameter from the request, which
fails because `Store` is not a valid Pydantic field type.
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.db.base import Store
from app.db.factory import get_store
from app.serving.service import RecommenderService

_service: RecommenderService | None = None


def store_dependency() -> Store:
    """The process-wide store."""
    return get_store()


def settings_dependency() -> Settings:
    return get_settings()


def service_dependency() -> RecommenderService:
    """The process-wide service wrapping the store and the model bundle."""
    global _service
    if _service is None:
        _service = RecommenderService(store_dependency())
    return _service


def get_service(store: Store | None = None) -> RecommenderService:
    """Programmatic accessor. Not a FastAPI dependency."""
    global _service
    if _service is None or (store is not None and _service.store is not store):
        _service = RecommenderService(store or store_dependency())
    return _service


def set_service(service: RecommenderService | None) -> None:
    """Override the singleton (used by tests and by scripts)."""
    global _service
    _service = service
