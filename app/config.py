"""Application settings, loaded from environment variables / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "Personalized Product Recommendation Engine"
    environment: str = Field(default="dev")
    api_prefix: str = "/api/v1"

    # ---- storage -------------------------------------------------------
    mongo_uri: str = Field(default="mongodb://localhost:27017")
    mongo_db: str = Field(default="reco")
    # When True the service starts even if no MongoDB is reachable and uses an
    # in-process store instead. Set to False for strict deployments.
    allow_memory_fallback: bool = Field(default=True)
    mongo_connect_timeout_ms: int = Field(default=1500)

    data_dir: Path = Field(default=BASE_DIR / "data")
    artifact_dir: Path = Field(default=BASE_DIR / "artifacts")

    # ---- synthetic data generator --------------------------------------
    n_users: int = Field(default=1500)
    n_products: int = Field(default=800)
    n_events: int = Field(default=120_000)
    random_seed: int = Field(default=42)

    # ---- modelling -----------------------------------------------------
    svd_factors: int = Field(default=64)
    knn_top_k: int = Field(default=50)
    content_top_k: int = Field(default=100)
    recency_halflife_days: float = Field(default=30.0)

    # ---- evaluation -----------------------------------------------------
    eval_top_k: tuple[int, ...] = (5, 10, 20)
    eval_holdout_per_user: int = Field(default=5)
    tuning_val_size: int = Field(default=2)
    # Hyperparameter search scores a random user sample rather than every user:
    # the ranking of a few hundred users is stable enough to pick weights, and it
    # turns a 30-minute grid search into a two-minute one.
    tuning_max_users: int = Field(default=300)
    tuning_seed: int = Field(default=7)
    # How much catalogue reach is worth relative to accuracy when choosing the
    # fusion weights. 0 = optimise NDCG only. Content-based models contribute
    # little raw accuracy but a large amount of coverage, so a pure-accuracy
    # objective will always drive the content weight to zero and silently throw
    # away the only strategy that can serve a brand-new product.
    tuning_coverage_weight: float = Field(default=0.05)

    # ---- serving --------------------------------------------------------
    default_top_k: int = Field(default=10)
    max_top_k: int = Field(default=100)
    model_cache_ttl_seconds: float = Field(default=60.0)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
