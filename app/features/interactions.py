"""Interaction-matrix construction.

Raw clickstream events are folded into one implicit user x item confidence
matrix. Two things happen here that materially change model quality:

1. **Event typing** - a view is not a purchase. `EVENT_WEIGHTS` scales each
   event type so implicit feedback reflects intent, not just traffic.
2. **Recency decay** - a product touched three months ago should matter less
   than one touched last week, so every weight is multiplied by an exponential
   half-life kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from app.domain import EVENT_WEIGHTS, POSITIVE_EVENTS


def event_weight(event_type: str, explicit: float | None = None) -> float:
    if explicit is not None and explicit > 0:
        return float(explicit)
    return EVENT_WEIGHTS.get(str(event_type), 1.0)


def recency_decay(
    timestamps: pd.Series, reference: pd.Timestamp, halflife_days: float
) -> np.ndarray:
    """Exponential half-life kernel, normalised so the newest event is ~1.0."""
    ts = pd.to_datetime(timestamps, utc=True, format="mixed")
    age_days = (reference - ts).dt.total_seconds().to_numpy() / 86400.0
    age_days = np.clip(age_days, 0.0, None)
    return np.power(0.5, age_days / max(halflife_days, 1e-6))


@dataclass
class InteractionMatrix:
    """CSR user x item implicit-feedback matrix plus the index maps."""

    matrix: sparse.csr_matrix
    user_ids: list[str] = field(default_factory=list)
    item_ids: list[str] = field(default_factory=list)
    reference_time: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        self._user_index = {u: i for i, u in enumerate(self.user_ids)}
        self._item_index = {p: i for i, p in enumerate(self.item_ids)}

    # -- shapes / indices -------------------------------------------------
    @property
    def n_users(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_items(self) -> int:
        return self.matrix.shape[1]

    def user_index(self, user_id: str) -> int | None:
        return self._user_index.get(user_id)

    def item_index(self, product_id: str) -> int | None:
        return self._item_index.get(product_id)

    def has_user(self, user_id: str) -> bool:
        return user_id in self._user_index

    # -- slicing ----------------------------------------------------------
    def user_row(self, user_id: str) -> sparse.csr_matrix:
        idx = self.user_index(user_id)
        if idx is None:
            raise KeyError(f"unknown user {user_id}")
        return self.matrix[idx]

    def seen_items(self, user_id: str) -> np.ndarray:
        idx = self.user_index(user_id)
        if idx is None:
            return np.array([], dtype=int)
        return self.matrix[idx].indices

    def user_ids_for_rows(self, rows: np.ndarray) -> list[str]:
        return [self.user_ids[i] for i in rows]

    def item_ids_for_cols(self, cols: np.ndarray) -> list[str]:
        return [self.item_ids[i] for i in cols]


def build_interaction_matrix(
    events: pd.DataFrame,
    product_ids: list[str],
    user_ids: list[str] | None = None,
    halflife_days: float = 30.0,
    positive_only: bool = False,
    reference_time: pd.Timestamp | None = None,
) -> InteractionMatrix:
    """Fold an event log into a user x item CSR matrix of decayed confidences."""
    if events.empty:
        return InteractionMatrix(
            sparse.csr_matrix((len(user_ids or []), len(product_ids))),
            list(user_ids or []),
            list(product_ids),
            reference_time,
        )

    frame = events.copy()
    frame["product_id"] = frame["product_id"].astype(str)
    frame["user_id"] = frame["user_id"].astype(str)

    if positive_only:
        frame = frame[frame["event_type"].isin(POSITIVE_EVENTS)]
        if frame.empty:
            return InteractionMatrix(
                sparse.csr_matrix((len(user_ids or []), len(product_ids))),
                list(user_ids or []),
                list(product_ids),
                reference_time,
            )

    if user_ids is None:
        user_ids = sorted(frame["user_id"].unique().tolist())
    else:
        user_ids = [u for u in user_ids if u in set(frame["user_id"])]

    reference = reference_time or pd.to_datetime(frame["timestamp"], utc=True, format="mixed").max()
    stored_weight = (
        frame["weight"]
        if "weight" in frame.columns
        else pd.Series([None] * len(frame), index=frame.index)
    )
    frame["_weight"] = [
        event_weight(t, w) for t, w in zip(frame["event_type"], stored_weight, strict=True)
    ]
    frame["_weight"] = frame["_weight"].astype(float)
    frame["_decay"] = recency_decay(frame["timestamp"], reference, halflife_days)
    frame["_value"] = frame["_weight"] * frame["_decay"]

    # Collapse repeat interactions on the same pair by summing.
    agg = (
        frame.groupby(["user_id", "product_id"], as_index=False)["_value"]
        .sum()
        .rename(columns={"_value": "value"})
    )

    user_index = {u: i for i, u in enumerate(user_ids)}
    item_index = {p: i for i, p in enumerate(product_ids)}
    agg = agg[
        agg["user_id"].isin(user_index) & agg["product_id"].isin(item_index)
    ]
    agg = agg.assign(
        u=agg["user_id"].map(user_index).astype(int),
        i=agg["product_id"].map(item_index).astype(int),
    )
    agg = agg[agg["value"] > 0]
    agg = agg.sort_values("value", ascending=False)

    matrix = sparse.coo_matrix(
        (agg["value"].to_numpy(float), (agg["u"].to_numpy(), agg["i"].to_numpy())),
        shape=(len(user_ids), len(product_ids)),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    # Log-dampen so a single heavy user cannot dominate a cosine column.
    matrix.data = np.log1p(matrix.data)
    matrix.eliminate_zeros()

    return InteractionMatrix(matrix, list(user_ids), list(product_ids), reference)


def interaction_frame(
    matrix: InteractionMatrix, events: pd.DataFrame, halflife_days: float = 30.0
) -> pd.DataFrame:
    """Dense-friendly (user_id, product_id, weight) view of the matrix."""
    coo = matrix.matrix.tocoo()
    reference = matrix.reference_time or pd.Timestamp.now(tz="UTC")
    return pd.DataFrame(
        {
            "user_id": matrix.user_ids_for_rows(coo.row),
            "product_id": matrix.item_ids_for_cols(coo.col),
            "value": coo.data,
        }
    ).assign(halflife_days=halflife_days, reference_time=reference)
