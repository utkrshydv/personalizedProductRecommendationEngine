"""Store factory: try MongoDB, fall back to the in-process store."""

from __future__ import annotations

import logging
import threading

from app.config import Settings, get_settings
from app.db.base import Store
from app.db.memory_store import MemoryStore
from app.db.mongo_store import MongoStore

log = logging.getLogger(__name__)

_lock = threading.Lock()
_store: Store | None = None


class StoreUnavailableError(RuntimeError):
    pass


def build_store(settings: Settings | None = None) -> Store:
    """Connect to MongoDB if possible, otherwise degrade to memory."""
    settings = settings or get_settings()
    try:
        store = MongoStore(
            settings.mongo_uri, settings.mongo_db, timeout_ms=settings.mongo_connect_timeout_ms
        )
        if store.ping():
            store.ensure_indexes()
            log.info("connected to MongoDB at %s (db=%s)", settings.mongo_uri, settings.mongo_db)
            return store
        store.close()
        log.warning("MongoDB at %s did not respond to ping", settings.mongo_uri)
    except Exception as exc:
        log.warning("MongoDB unavailable (%s): %s", settings.mongo_uri, exc)

    if not settings.allow_memory_fallback:
        raise StoreUnavailableError(
            f"MongoDB at {settings.mongo_uri} is unreachable and memory fallback is disabled"
        )
    log.warning("Falling back to the in-memory store; data will not persist.")
    return MemoryStore()


def get_store(settings: Settings | None = None) -> Store:
    """Process-wide singleton store."""
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = build_store(settings)
    return _store


def set_store(store: Store | None) -> None:
    """Override the singleton (used by tests and by scripts)."""
    global _store
    with _lock:
        if _store is not None and _store is not store:
            _store.close()
        _store = store


def reset_store() -> None:
    set_store(None)
