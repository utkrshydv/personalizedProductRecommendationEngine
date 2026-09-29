"""Recommendation, catalogue and interaction endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from starlette.concurrency import run_in_threadpool

from app.deps import service_dependency
from app.schemas import (
    EventIn,
    EventOut,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
    SimilarRequest,
    SimilarResponse,
    UserProfile,
)
from app.serving.service import ModelNotReadyError, RecommenderService

log = logging.getLogger(__name__)
router = APIRouter(tags=["recommendations"])


@router.post(
    "/recommendations",
    response_model=RecommendationResponse,
    summary="Personalised recommendations for a user",
)
async def create_recommendations(
    request: RecommendationRequest,
    service: RecommenderService = Depends(service_dependency),
) -> RecommendationResponse:
    """Rank products for one user.

    The blocking work (sparse matrix products, top-k selection) runs in a
    threadpool so the event loop stays free to serve other users.
    """
    try:
        return await run_in_threadpool(service.recommend, request)
    except ModelNotReadyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unknown strategy {exc.args[0]!r}; "
            f"available: {', '.join(service.bundle().strategies)}",
        ) from exc


@router.get(
    "/users/{user_id}/recommendations",
    response_model=RecommendationResponse,
    summary="Recommendations for a user (convenience GET)",
)
async def get_recommendations(
    user_id: str,
    top_k: int = Query(default=10, ge=1, le=100),
    strategy: str = Query(default="hybrid"),
    include_seen: bool = Query(default=False),
    service: RecommenderService = Depends(service_dependency),
) -> RecommendationResponse:
    return await create_recommendations(
        RecommendationRequest(
            user_id=user_id, top_k=top_k, strategy=strategy, include_seen=include_seen
        ),
        service,
    )


@router.post(
    "/products/{product_id}/similar",
    response_model=SimilarResponse,
    summary="Item-to-item recommendations ('more like this')",
)
async def similar_products(
    product_id: str,
    request: SimilarRequest,
    service: RecommenderService = Depends(service_dependency),
) -> SimilarResponse:
    try:
        bundle = service.bundle()
        if bundle is None:
            raise ModelNotReadyError("no trained model available")
        top_k = request.top_k or 10
        items = await run_in_threadpool(service.similar, product_id, top_k, request.strategy)
    except KeyError:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"product {product_id!r} not found"
        ) from None
    except ModelNotReadyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return SimilarResponse(
        product_id=product_id, strategy=request.strategy, items=items,
        model_version=bundle.version,
    )


@router.post(
    "/events",
    response_model=EventOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Record an interaction (real-time feedback loop)",
)
async def record_event(
    event: EventIn,
    service: RecommenderService = Depends(service_dependency),
) -> EventOut:
    """Persist an interaction and update the live user profile in place.

    Accepting a `view` costs a write and no model work; a `purchase` or
    `add_to_cart` also folds the item into the user's content profile so the
    next recommendation request already reflects it.
    """
    try:
        stored = await run_in_threadpool(
            service.record_event,
            event.user_id,
            event.product_id,
            event.event_type.value,
            event.session_id,
            event.timestamp,
            event.weight,
        )
    except Exception as exc:
        log.exception("failed to record event")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"could not record event: {exc}"
        ) from exc
    return EventOut(
        event_id=stored.event_id,
        user_id=stored.user_id,
        product_id=stored.product_id,
        event_type=stored.event_type,
        timestamp=stored.timestamp,
        accepted=True,
        profile_updated=event.event_type.value != "view",
    )


@router.get("/users/{user_id}/profile", response_model=UserProfile, summary="User profile")
async def user_profile(
    user_id: str, service: RecommenderService = Depends(service_dependency)
) -> UserProfile:
    return await run_in_threadpool(service.profile, user_id)


@router.get("/trending", response_model=list[RecommendationItem], summary="Global trending")
async def trending(
    top_k: int = Query(default=10, ge=1, le=100),
    service: RecommenderService = Depends(service_dependency),
) -> list[RecommendationItem]:
    try:
        return await run_in_threadpool(service.trending, top_k)
    except ModelNotReadyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
