#!/usr/bin/env python
"""Compare every strategy offline, without touching the stored model.

    python -m scripts.evaluate
    python -m scripts.evaluate --segment cold_start --top-k 10
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from app.config import get_settings
from app.data.synthetic import generate_dataset
from app.db.factory import get_store
from app.eval.evaluate import evaluate_all, results_to_frame
from app.training.pipeline import ingest

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("evaluate")

COLUMNS = [
    "segment", "model", "k", "n_users", "precision", "recall", "map",
    "ndcg", "mrr", "hit_rate", "coverage", "novelty", "personalisation", "diversity",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--segment", default=None,
                        choices=["all", "cold_start", "warm"],
                        help="restrict the report to one user segment")
    parser.add_argument("--seed-data", action="store_true",
                        help="generate a fresh dataset instead of reading the store")
    args = parser.parse_args(argv)

    settings = get_settings()

    if args.seed_data:
        frames = generate_dataset(settings)
        products, events = frames["products"], frames["events"]
    else:
        store = get_store(settings)
        if store.products().empty:
            log.info("store is empty; generating a dataset")
            ingest(store, settings, generate=True)
        products, events = store.products(), store.events()

    results, split = evaluate_all(events, products, settings)
    frame = results_to_frame(results)
    frame = frame[frame["k"] == args.top_k]
    if args.segment:
        frame = frame[frame["segment"] == args.segment]

    show = [c for c in COLUMNS if c in frame.columns]
    pd.set_option("display.width", 200)
    print(f"\ntrain={len(split.train_events)} val={len(split.val_events)} "
          f"test={len(split.test_events)} users={split.n_users}")
    print(f"\n=== offline evaluation @ k={args.top_k} ===")
    print(frame[show].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
