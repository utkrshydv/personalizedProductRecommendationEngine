"""Health and administrative endpoints (retrain, metrics, model info)."""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.deps import service_dependency, settings_dependency
from app.schemas import (
    HealthResponse,
    MetricsResponse,
    ModelInfo,
    TrainRequest,
    TrainResponse,
)
from app.serving.service import RecommenderService
from app.training.artifacts import load_metrics, load_tuning, utcnow

log = logging.getLogger(__name__)
router = APIRouter(tags=["admin"])

_train_state: dict = {"status": "idle", "message": "", "version": None}
_train_lock = threading.Lock()


@router.get("/health", response_model=HealthResponse, summary="Liveness and readiness")
async def health(
    service: RecommenderService = Depends(service_dependency),
    settings: Settings = Depends(settings_dependency),
) -> HealthResponse:
    status_info = service.status()
    ready = bool(status_info["model_version"])
    return HealthResponse(
        status="ok" if ready else "degraded",
        store_backend=status_info["store_backend"],
        model_version=status_info["model_version"],
        models_loaded=status_info["models_loaded"],
        counts=status_info["counts"],
        environment=settings.environment,
    )


@router.get("/models", response_model=list[ModelInfo], summary="Loaded model versions")
async def models(service: RecommenderService = Depends(service_dependency)) -> list[ModelInfo]:
    bundle = service.bundle()
    if bundle is None:
        return []
    interactions = bundle.interactions
    return [
        ModelInfo(
            name=strategy,
            strategy=strategy,
            version=bundle.version,
            trained_at=bundle.trained_at,
            n_users=interactions.n_users if interactions else None,
            n_items=interactions.n_items if interactions else None,
            params=bundle.params,
        )
        for strategy in bundle.strategies
    ]


@router.get("/metrics", response_model=MetricsResponse, summary="Latest offline metrics")
async def metrics(settings: Settings = Depends(settings_dependency)) -> MetricsResponse:
    rows = load_metrics(settings)
    return MetricsResponse(
        model_version=rows[0].get("model_version", "unknown") if rows else "unknown",
        generated_at=utcnow(),
        top_k=list(settings.eval_top_k),
        rows=rows,
    )


@router.get("/tuning", summary="Fusion-weight search results")
async def tuning(settings: Settings = Depends(settings_dependency)) -> dict:
    return load_tuning(settings) or {"message": "no tuning run recorded yet"}


@router.get("/admin/train/status", summary="Retrain progress")
async def train_status() -> dict:
    with _train_lock:
        return dict(_train_state)


@router.post("/admin/train", response_model=TrainResponse, summary="Run the training pipeline")
async def trigger_train(
    request: TrainRequest,
    service: RecommenderService = Depends(service_dependency),
) -> TrainResponse:
    """Regenerate data (optional), retune, refit and re-evaluate.

    Runs in a background thread by default so the API stays responsive; the
    previously loaded bundle keeps serving traffic until the new one is
    complete, which is the whole point of writing the artifact atomically.
    """
    if not _train_lock.acquire(blocking=False):
        return TrainResponse(
            status="started", message="a training run is already in progress", model_version=None
        )
    _train_lock.release()

    def _run() -> None:
        from app.training.pipeline import ingest, train

        with _train_lock:
            _train_state.update(status="running", message="ingesting data", version=None)
        try:
            if request.generate_data:
                ingest(service.store, get_settings(), generate=True)
            with _train_lock:
                _train_state.update(status="running", message="tuning and fitting")
            bundle, _frame = train(
                service.store, get_settings(), tune_weights=request.tune_weights
            )
            service.reload()
            with _train_lock:
                _train_state.update(
                    status="completed", message="done", version=bundle.version
                )
            log.info("retrained to version %s", bundle.version)
        except Exception as exc:
            log.exception("training failed")
            with _train_lock:
                _train_state.update(status="failed", message=str(exc), version=None)

    if request.background:
        threading.Thread(target=_run, name="reco-trainer", daemon=True).start()
        return TrainResponse(
            status="started", message="training started in the background",
            model_version=service.model_version,
        )

    await run_in_threadpool(_run)
    with _train_lock:
        state = dict(_train_state)
    return TrainResponse(
        status=state["status"], message=state["message"], model_version=state["version"]
    )


@router.post("/admin/reload", summary="Reload the model bundle from disk")
async def reload_models(service: RecommenderService = Depends(service_dependency)) -> dict:
    ok = await run_in_threadpool(service.reload)
    if not ok:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "no trained model bundle on disk"
        )
    return {"reloaded": True, "model_version": service.model_version}
