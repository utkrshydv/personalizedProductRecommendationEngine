"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.domain import EventType

Strategy = Literal["hybrid", "collaborative", "content", "popularity"]


class ProductOut(BaseModel):
    product_id: str
    title: str
    description: str
    category: str
    subcategory: str
    brand: str
    price: float
    rating: float
    rating_count: int
    tags: list[str] = Field(default_factory=list)
    popularity: float | None = None
    score: float | None = None
    rank: int | None = None
    reason: str | None = None


class RecommendationItem(ProductOut):
    """A product plus the fields a ranked response needs."""


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., min_length=1, examples=["U00042"])
    top_k: int | None = Field(default=None, ge=1, description="Defaults to 10.")
    strategy: Strategy = "hybrid"
    include_seen: bool = False
    exclude_product_ids: list[str] = Field(default_factory=list)
    category: str | None = None
    brand: str | None = None
    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)
    tags: list[str] = Field(default_factory=list)

    @field_validator("max_price")
    @classmethod
    def _positive_max(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("max_price must be non-negative")
        return v


class RecommendationResponse(BaseModel):
    user_id: str
    strategy: Strategy
    top_k: int
    items: list[RecommendationItem]
    count: int
    model_version: str
    generated_at: datetime
    cold_start: bool = False
    latency_ms: float = 0.0


class SimilarRequest(BaseModel):
    top_k: int | None = Field(default=None, ge=1)
    strategy: Strategy = "hybrid"


class SimilarResponse(BaseModel):
    product_id: str
    strategy: Strategy
    items: list[RecommendationItem]
    model_version: str


class EventIn(BaseModel):
    user_id: str = Field(..., min_length=1)
    product_id: str = Field(..., min_length=1)
    event_type: EventType = EventType.VIEW
    session_id: str | None = None
    timestamp: datetime | None = None
    weight: float | None = Field(default=None, gt=0)

    @field_validator("timestamp")
    @classmethod
    def _tz_aware(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            raise ValueError("timestamp must include a timezone offset")
        return v


class EventOut(BaseModel):
    event_id: str
    user_id: str
    product_id: str
    event_type: str
    timestamp: datetime
    accepted: bool = True
    profile_updated: bool = False


class UserProfile(BaseModel):
    user_id: str
    exists: bool
    n_events: int = 0
    n_distinct_products: int = 0
    event_type_counts: dict[str, int] = Field(default_factory=dict)
    top_categories: list[tuple[str, int]] = Field(default_factory=list)
    top_brands: list[tuple[str, int]] = Field(default_factory=list)
    price_range: tuple[float, float] | None = None
    cold_start: bool = True


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    store_backend: str
    model_version: str | None
    models_loaded: list[str]
    counts: dict[str, int]
    environment: str


class ModelInfo(BaseModel):
    name: str
    strategy: str
    version: str
    trained_at: datetime | None = None
    n_users: int | None = None
    n_items: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    model_version: str
    generated_at: datetime
    top_k: list[int]
    rows: list[dict[str, Any]]


class TrainRequest(BaseModel):
    generate_data: bool = Field(
        default=False, description="Regenerate the synthetic dataset before training."
    )
    tune_weights: bool = Field(
        default=True, description="Grid-search fusion weights on the validation slice."
    )
    background: bool = True


class TrainResponse(BaseModel):
    status: Literal["started", "completed", "failed"]
    message: str
    model_version: str | None = None
