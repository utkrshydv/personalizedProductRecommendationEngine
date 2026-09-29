"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.api import admin, products, recommendations
from app.config import get_settings
from app.db.base import Store
from app.db.factory import get_store
from app.deps import get_service
from app.serving.service import ModelNotReadyError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def create_app(store: Store | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved = store or get_store()
        log.info("store backend: %s", resolved.backend)
        if store is not None:
            get_service(resolved)
        # Warm the bundle off-thread so the first request is not the one that
        # pays for deserialising the model.
        try:
            service = get_service()
            await run_in_threadpool(service.bundle)
            log.info("model version: %s", service.model_version)
        except Exception as exc:
            log.warning("model bundle unavailable at boot: %s", exc)
        yield
        if store is None:
            resolved.close()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description=(
            "Personalised product recommendations built on user interactions, "
            "product metadata and browsing history. Content-based, collaborative "
            "and hybrid strategies, compared with offline ranking metrics."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(admin.router, prefix=settings.api_prefix)
    app.include_router(recommendations.router, prefix=settings.api_prefix)
    app.include_router(products.router, prefix=settings.api_prefix)

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {
            "service": settings.app_name,
            "docs": "/docs",
            "health": f"{settings.api_prefix}/health",
        }

    @app.exception_handler(ModelNotReadyError)
    async def model_not_ready(_: Request, exc: ModelNotReadyError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "detail": str(exc),
                "hint": "POST /api/v1/admin/train with generate_data=true to bootstrap",
            },
        )

    return app


app = create_app()
