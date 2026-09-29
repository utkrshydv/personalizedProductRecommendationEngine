#!/usr/bin/env python
"""Exercise the API end to end against a running server.

    python -m scripts.smoke_test
    python -m scripts.smoke_test --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def call(base: str, path: str, payload: dict | None = None) -> tuple[int, object]:
    url = f"{base}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()[:300]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", default="U00042")
    args = parser.parse_args(argv)
    base = args.base_url.rstrip("/")
    prefix = "/api/v1"
    failures = 0

    checks: list[tuple[str, str, dict | None]] = [
        ("health", f"{prefix}/health", None),
        ("facets", f"{prefix}/products/facets", None),
        ("products", f"{prefix}/products?limit=3", None),
        ("recommend", f"{prefix}/recommendations",
         {"user_id": args.user_id, "top_k": 5, "strategy": "hybrid"}),
        ("recommend+filter", f"{prefix}/recommendations",
         {"user_id": args.user_id, "top_k": 5, "strategy": "content", "max_price": 60}),
        ("profile", f"{prefix}/users/{args.user_id}/profile", None),
        ("trending", f"{prefix}/trending?top_k=3", None),
        ("metrics", f"{prefix}/metrics", None),
    ]

    for name, path, payload in checks:
        code, body = call(base, path, payload)
        ok = code == 200
        failures += 0 if ok else 1
        summary = ""
        if isinstance(body, dict):
            summary = f"keys={sorted(body)[:6]}"
            items = body.get("items")
            if isinstance(items, list) and items:
                summary += f" top={items[0].get('title', '?')[:40]!r} score={items[0].get('score')}"
        print(f"[{'ok ' if ok else 'FAIL'}] {name:16s} {code} {summary}")

    # Similar-items needs a real product id from the catalogue.
    code, body = call(base, f"{prefix}/products?limit=1")
    if code == 200 and isinstance(body, list) and body:
        pid = body[0]["product_id"]
        code, payload = call(base, f"{prefix}/products/{pid}/similar", {"top_k": 5})
        ok = code == 200
        failures += 0 if ok else 1
        has_items = isinstance(payload, dict) and payload.get("items")
        top = payload["items"][0]["title"][:40] if has_items else "-"
        print(f"[{'ok ' if ok else 'FAIL'}] {'similar':16s} {code} top={top!r}")

    # Event ingestion.
    code, payload = call(base, f"{prefix}/events",
                         {"user_id": args.user_id, "product_id": pid,
                          "event_type": "add_to_cart"})
    ok = code == 202
    failures += 0 if ok else 1
    print(f"[{'ok ' if ok else 'FAIL'}] {'event':16s} {code} {payload if not ok else 'accepted'}")

    print(f"\n{'all checks passed' if failures == 0 else f'{failures} check(s) failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
