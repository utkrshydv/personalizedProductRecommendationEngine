"""Model persistence.

A single joblib bundle holds everything needed to serve: the fitted feature
builder, the interaction matrix, every trained strategy, and the product
metadata used to hydrate responses. Artifacts are versioned by content
fingerprint so a stale bundle can never be served after a retrain.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.config import Settings, get_settings

log = logging.getLogger(__name__)

BUNDLE_NAME = "model_bundle.joblib"
METRICS_NAME = "eval_metrics.json"
META_NAME = "model_meta.json"


def data_fingerprint(products: pd.DataFrame, events: pd.DataFrame) -> str:
    """Short, stable hash of the training data, used as the model version."""
    parts = [
        f"products={len(products)}",
        f"events={len(events)}",
        f"max_ts={events['timestamp'].max() if 'timestamp' in events else ''}",
        f"cols={','.join(sorted(products.columns))}",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


@dataclass
class ModelBundle:
    version: str
    trained_at: datetime
    models: dict[str, Any] = field(default_factory=dict)
    features: Any = None
    interactions: Any = None
    products: pd.DataFrame = field(default_factory=pd.DataFrame)
    item_matrix: Any = None
    params: dict[str, Any] = field(default_factory=dict)
    tuning: dict[str, Any] = field(default_factory=dict)

    def model(self, name: str):
        if name not in self.models:
            raise KeyError(
                f"strategy {name!r} not loaded; available: {sorted(self.models)}"
            )
        return self.models[name]

    @property
    def strategies(self) -> list[str]:
        return sorted(self.models)

    def summary(self) -> dict[str, Any]:
        interactions = self.interactions
        return {
            "version": self.version,
            "trained_at": self.trained_at.isoformat(),
            "strategies": self.strategies,
            "n_users": interactions.n_users if interactions else 0,
            "n_items": interactions.n_items if interactions else 0,
            "params": self.params,
            "tuning": self.tuning,
        }


def artifact_paths(settings: Settings | None = None) -> dict[str, Path]:
    settings = settings or get_settings()
    root = settings.artifact_dir
    return {
        "bundle": root / BUNDLE_NAME,
        "metrics": root / METRICS_NAME,
        "meta": root / META_NAME,
    }


def save_bundle(
    bundle: ModelBundle,
    metrics: list[dict] | None = None,
    settings: Settings | None = None,
) -> dict[str, Path]:
    settings = settings or get_settings()
    paths = artifact_paths(settings)
    paths["bundle"].parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file then swap, so a crash mid-write cannot leave a
    # half-written bundle that the API would fail to load on boot.
    tmp = paths["bundle"].with_suffix(".tmp")
    joblib.dump(bundle, tmp, compress=3)
    tmp.replace(paths["bundle"])

    paths["meta"].write_text(json.dumps(bundle.summary(), indent=2, default=str), encoding="utf-8")
    if metrics is not None:
        paths["metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return paths


def load_bundle(settings: Settings | None = None) -> ModelBundle | None:
    settings = settings or get_settings()
    path = artifact_paths(settings)["bundle"]
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        log.error("failed to load model bundle from %s: %s", path, exc)
        return None
    if not isinstance(bundle, ModelBundle):
        log.error("artifact at %s is not a ModelBundle", path)
        return None
    return bundle


def load_metrics(settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    path = artifact_paths(settings)["metrics"]
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def load_tuning(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    path = settings.artifact_dir / "fusion_weights.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
