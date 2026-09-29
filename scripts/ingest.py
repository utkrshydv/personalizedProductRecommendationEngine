#!/usr/bin/env python
"""Generate the synthetic dataset and load it into the store.

    python -m scripts.ingest --generate         # generate + load (replaces contents)
    python -m scripts.ingest                     # no-op: data is already loaded
    python -m scripts.ingest --generate --keep  # append instead of replacing
    python -m scripts.ingest --stats-only        # just show what is in the store

Re-ingesting without --generate never clears the store, so the command is safe
to re-run against existing data.
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.config import get_settings
from app.db.factory import get_store
from app.training.pipeline import ingest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("ingest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true",
                        help="generate the synthetic dataset (required on an empty store)")
    parser.add_argument("--keep", action="store_true",
                        help="append instead of clearing the store first")
    parser.add_argument("--stats-only", action="store_true", help="print store counts and exit")
    args = parser.parse_args(argv)

    settings = get_settings()
    store = get_store(settings)
    log.info("store backend: %s", store.backend)

    if args.stats_only:
        print(store.counts())
        return 0

    if not args.generate and store.counts()["events"] == 0:
        parser.error(
            "the store is empty; re-run with --generate to seed it with the "
            "synthetic dataset (or load your own data first)"
        )

    # Only wipe the store when we are about to replace it with a freshly
    # generated dataset. Clearing on the re-ingest path would destroy exactly
    # the data the command was asked to load.
    clear = args.generate and not args.keep
    report = ingest(store, settings, generate=args.generate, clear=clear)
    print(
        f"ingested {report.products} products / {report.users} users / "
        f"{report.events} events from {report.source} in {report.seconds:.2f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
