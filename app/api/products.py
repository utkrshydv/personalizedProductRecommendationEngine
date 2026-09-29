"""Catalogue browsing endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from starlette.concurrency import run_in_threadpool

from app.deps import service_dependency
from app.schemas import ProductOut
from app.serving.service import ModelNotReadyError, RecommenderService

router = APIRouter(prefix="/products", tags=["catalogue"])


@router.get("", response_model=list[ProductOut], summary="Browse the catalogue")
async def list_products(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    category: str | None = None,
    brand: str | None = None,
    search: str | None = Query(default=None, max_length=120),
    service: RecommenderService = Depends(service_dependency),
) -> list[ProductOut]:
    try:
        return await run_in_threadpool(
            service.products, limit, offset, category, brand, search
        )
    except ModelNotReadyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@router.get("/facets", summary="Available filters")
async def facets(service: RecommenderService = Depends(service_dependency)) -> dict:
    try:
        return await run_in_threadpool(service.facets)
    except ModelNotReadyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@router.get("/{product_id}", response_model=ProductOut, summary="Fetch one product")
async def get_product(
    product_id: str, service: RecommenderService = Depends(service_dependency)
) -> ProductOut:
    try:
        product = await run_in_threadpool(service.product, product_id)
    except ModelNotReadyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    if product is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"product {product_id!r} not found")
    return product
