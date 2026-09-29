#!/usr/bin/env python
"""Run the full training pipeline and persist the model bundle.

    python -m scripts.train                 # tune, fit, evaluate, save
    python -m scripts.train --no-tune       # skip the weight search
    python -m scripts.train --generate      # regenerate data first
    python -m scripts.train --report        # print the metrics table and exit
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from app.config import get_settings
from app.db.factory import get_store
from app.training.artifacts import load_metrics
from app.training.pipeline import ingest, train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("train")

COLUMNS = [
    "segment", "model", "k", "n_users", "precision", "recall", "map",
    "ndcg", "mrr", "hit_rate", "coverage", "novelty", "personalisation",
]


def _flatten(rows: list[dict]) -> pd.DataFrame:
    """Stored rows keep metrics nested under "metrics"; the report wants them flat."""
    flat = [
        {k: v for k, v in row.items() if k != "metrics"} | dict(row.get("metrics", {}))
        for row in rows
    ]
    return pd.DataFrame(flat)


def _print_report(rows: list[dict], k: int = 10) -> None:
    if not rows:
        print("no metrics recorded yet")
        return
    frame = _flatten(rows)
    frame = frame[frame["k"] == k]
    if frame.empty:
        return
    show = [c for c in COLUMNS if c in frame.columns]
    numeric = [c for c in show if c not in {"segment", "model"}]
    frame = frame[show].sort_values("ndcg", ascending=False)
    print(f"\n=== offline evaluation @ k={k} ===")
    print(frame.to_string(index=False, formatters={c: lambda v: f"{v:.4f}" for c in numeric}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="regenerate the dataset first")
    parser.add_argument("--no-tune", action="store_true", help="skip the fusion-weight search")
    parser.add_argument("--report", action="store_true", help="print stored metrics and exit")
    parser.add_argument("--top-k", type=int, default=10, help="k for the printed report")
    args = parser.parse_args(argv)

    settings = get_settings()
    store = get_store(settings)
    log.info("store backend: %s", store.backend)

    if args.report:
        _print_report(load_metrics(settings), args.top_k)
        return 0

    if args.generate:
        ingest(store, settings, generate=True)

    bundle, frame = train(store, settings, tune_weights=not args.no_tune)
    _print_report(frame.to_dict("records"), args.top_k)

    print(f"\nmodel version : {bundle.version}")
    print(f"trained at    : {bundle.trained_at.isoformat()}")
    print(f"strategies    : {', '.join(bundle.strategies)}")
    print(f"catalog       : {bundle.interactions.n_items} items x "
          f"{bundle.interactions.n_users} users")
    print(f"artifacts     : {settings.artifact_dir}")
    if bundle.tuning:
        weights = ", ".join(
            f"{k}={bundle.params.get(k)}"
            for k in ("content_weight", "collab_weight", "knn_weight")
            if bundle.params.get(k) is not None
        )
        print(f"fusion weights: {weights}")
        print(f"objective     : {bundle.tuning.get('objective')}")
        print(
            f"vs collaborative-only on validation: "
            f"NDCG@10 {bundle.tuning.get('ndcg_lift_over_collaborative')}, "
            f"coverage {bundle.tuning.get('coverage_lift_over_collaborative')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
