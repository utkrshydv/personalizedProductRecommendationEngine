"""Core domain records shared by the pipeline, the store and the models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EventType(str, Enum):
    VIEW = "view"
    ADD_TO_CART = "add_to_cart"
    WISHLIST = "wishlist"
    PURCHASE = "purchase"
    RATING = "rating"


#: How strongly each interaction type signals preference. Used to turn raw
#: events into an implicit-feedback confidence value.
EVENT_WEIGHTS: dict[str, float] = {
    EventType.VIEW.value: 1.0,
    EventType.WISHLIST.value: 2.0,
    EventType.ADD_TO_CART.value: 3.0,
    EventType.RATING.value: 4.0,
    EventType.PURCHASE.value: 5.0,
}

#: Event types that count as a positive signal during offline evaluation.
POSITIVE_EVENTS: frozenset[str] = frozenset(
    {EventType.PURCHASE.value, EventType.ADD_TO_CART.value, EventType.WISHLIST.value}
)


@dataclass(slots=True)
class Product:
    product_id: str
    title: str
    description: str
    category: str
    subcategory: str
    brand: str
    price: float
    rating: float
    rating_count: int
    tags: list[str] = field(default_factory=list)
    created_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if isinstance(self.created_at, datetime):
            payload["created_at"] = self.created_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Product:
        data = {k: raw.get(k) for k in cls.__slots__}  # type: ignore[attr-defined]
        data.setdefault("tags", [])
        created = raw.get("created_at")
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        if created is not None and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        data["created_at"] = created
        return cls(**data)  # type: ignore[arg-type]


@dataclass(slots=True)
class User:
    user_id: str
    country: str = "US"
    age_band: str = "25-34"
    created_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if isinstance(self.created_at, datetime):
            payload["created_at"] = self.created_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> User:
        created = raw.get("created_at")
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        if created is not None and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return cls(
            user_id=raw["user_id"],
            country=raw.get("country", "US"),
            age_band=raw.get("age_band", "25-34"),
            created_at=created,
        )


@dataclass(slots=True)
class Event:
    event_id: str
    user_id: str
    product_id: str
    event_type: str
    timestamp: datetime
    weight: float = 1.0
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "product_id": self.product_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "weight": self.weight,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Event:
        ts = raw["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return cls(
            event_id=raw["event_id"],
            user_id=raw["user_id"],
            product_id=raw["product_id"],
            event_type=raw.get("event_type", EventType.VIEW.value),
            timestamp=ts,
            weight=float(raw.get("weight", 1.0)),
            session_id=raw.get("session_id"),
        )
