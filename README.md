# Personalised Product Recommendation Engine

Ranks products from **user interactions, product metadata and browsing
history**, using four strategies that are compared against each other with
offline ranking metrics rather than by assertion.

```
Python 3.10+ · FastAPI · Scikit-learn · Pandas · SciPy · MongoDB (pymongo)
```

---

## In plain terms

> This section is the conceptual overview — what problem is being solved, why
> there are four approaches, and how any of it is verified. Everything below it
> is the technical detail: architecture, model internals, metrics, and the full
> results table.

### The problem

Consider an online store. It holds 800 products, and it holds a log of what each
shopper has done: viewed, added to basket, wishlisted, bought.

Someone buys **Nike Air Running Shoes — breathable, blue**.

The question the system answers is: *of these 800 products, which other ten
should this person be shown?*

This is not a lookup. It is a judgement, made 800-to-10, and the answer is
worth money in both directions: showing the right ten converts, showing the same
ten popular products to everyone wastes the catalogue and eventually starves the
long tail of products nobody ever sees.

There are two fundamentally different ways to make that judgement, and this
project implements both.

### The two ways of guessing

#### 1. Compare the products themselves

The shoes are breathable, blue, running shoes, and priced within a range the
shopper has historically bought from. So show them more breathable blue running
shoes.

Formally: every product is converted into a vector of numbers derived from its
text and attributes, and the products most similar to what the shopper already
bought are surfaced. Called **content-based filtering**.

Its strength: **it works on a product nobody has bought yet.** A newly listed
item has no purchase history, but its description, category, brand and price are
known immediately. This method can rank it on day one.

#### 2. Compare the people

Four hundred other shoppers who bought those shoes also bought a particular
water bottle. So this person probably wants the water bottle.

Formally: products that repeatedly appear together across many shoppers are
treated as related, and shoppers are shown what related products other similar
shoppers chose. Called **collaborative filtering**.

Its strength: **it is more accurate**, because observed behaviour beats
description. A product's text is written by a marketer; its purchase history is
not.

Its weakness: a product that launched yesterday has no history, so this method
knows nothing about it at all.

#### 3. The deliberately unsophisticated one

Show everyone the most popular products, ignoring who they are.

This is included on purpose. It is a **control**, and it is the fastest way to
detect a personalised model that has quietly degenerated into something that
just ranks by date or by recency. If a sophisticated model cannot beat "most
popular", it is not personalising anything, and the fact would otherwise be
invisible in the output.

#### 4. The combination

The first method is strong where the second is blind; the second is strong
where the first is merely adequate. The service runs both and fuses their
rankings into a single list, so the result inherits the accuracy of behaviour
without giving up the long-tail reach of product description.

Because these two methods fail in *different* circumstances rather than one
dominating the other, combining them is worth doing, and it is the strategy the
service uses by default.

### How we know whether any of it works

A recommender is easy to build and easy to fool yourself about. This is the part
of the project that matters most, so it is worth being precise about.

**The method.** Take each shopper's most recent few purchases and hide them.
Build the model using everything else. Then ask a direct question: did the model
place those hidden purchases inside its top ten? Repeat across every shopper and
average. That is a score, and it can be computed for each of the four strategies
and printed side by side.

**Why the recent purchases, and not a random few.** This is the single most
important subtlety in the whole project. If a random handful of purchases is
hidden, the model can still see the future: the shopper browsed the same
product two days *after* the hidden purchase, and that browsing is still in the
training data. The model effectively memorised the answer before being tested on
it. Every score comes out inflated, and nothing in the output looks wrong.

The fix is to cut each shopper's training data at the point where their hidden
purchases begin, so the model only ever sees the past. This was implemented
incorrectly in the first version of this project, and it is now a test that
fails if the behaviour ever regresses.

**Why the results are split by shopper type.** Averaging everyone into one number
destroys the most useful information. Shoppers with a long history and shoppers
with almost none are different problems, and the best method for one is often not
the best method for the other. When averaged, their opposite results cancel and
the two leading methods look like a tie.

Reported separately, it becomes clear that the combined strategy genuinely wins
for near-new shoppers, while the purely behavioural method is better for
established ones. The blended average would have concealed exactly the finding
that justifies building a combined system at all.

### The live service

The engine is exposed as a web service. A request asks for ten products for a
given shopper and returns a ranked list in a few milliseconds.

One design detail is worth calling out, because it is the clearest practical
difference between the two main methods. When a shopper buys something, the
content-based model updates **immediately**, with no retraining: its profile is
a running average of everything that shopper has interacted with, so recording
a new purchase is literally adding one more term to an average. The
collaborative model cannot do this — its picture of the shopper is baked into a
matrix computed at training time, and it stays stale until the whole model is
rebuilt.

That is the real reason both methods are kept. The first is cheap and immediate
but approximate; the second is accurate but needs periodic rebuilding. Used
together, each covers the other's blind spot.

### What this project does not have

Stated up front, because the numbers below are easy to over-read.

**The store is invented.** No real product or customer data was available, so
the catalogue and the entire shopping history are generated. They are generated
with deliberate realism — each shopper has consistent preferences, prices follow
a realistic distribution, and popular products genuinely attract more shoppers —
so the *comparison between the four strategies* is meaningful and the evaluation
methodology transfers to real data unchanged. The *absolute scores do not*, and
should not be quoted as a performance claim. They indicate that the pipeline is
built correctly, not that a real shop would see these results.

Also absent: any way to load real product data, authentication and rate
limiting, and any online or A/B evaluation. This is research code behind a
production-shaped interface, not a service to point at customers. The full list
is in [Honest limitations](#honest-limitations).

---

## Technical detail

The remainder of this document covers implementation: the architecture, how each
model is constructed, the evaluation harness, measured results, the API surface,
and the design trade-offs taken.

- [Quick start](#quick-start)
- [Architecture](#architecture)
- [The models](#the-models)
- [Evaluation](#evaluation)
- [Results](#results)
- [API](#api)
- [Configuration](#configuration)
- [Testing](#testing)
- [Design notes](#design-notes)
- [Honest limitations](#honest-limitations)

---

### Quick start

```bash
pip install -r requirements.txt

# 1. Generate the dataset and load it into MongoDB
python -m scripts.ingest --generate

# 2. Tune fusion weights, fit every model, evaluate, save artifacts
python -m scripts.train

# 3. Serve
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs>.

```bash
python -m scripts.evaluate --top-k 10     # compare strategies offline
python -m scripts.train --report         # reprint the stored metrics table
python -m scripts.smoke_test             # exercise a running server end to end
python -m pytest -q                      # 184 tests
ruff check app scripts tests             # clean
```

**Runtime:** ~18s to generate the dataset, ~20 min for a full training run
(of which ~19 min is the 63-point weight grid; `--no-tune` takes ~90s).

> **No MongoDB available?** The service starts anyway on an in-process store and
> logs `Falling back to the in-memory store`. Set
> `ALLOW_MEMORY_FALLBACK=false` to make that a hard startup failure instead.
> MongoDB was available during development, so the `MongoStore` path is
> exercised; see [Testing](#testing) for how.

---

### Architecture

```
             ┌──────────────┐
traffic ──▶ | │  FastAPI  │ | app/api
             └──────┬───────┘
                    │  blocking work → threadpool
             ┌──────▼────────────────┐
             │ RecommenderService    │  app/serving
             │  · bundle (TTL cache) │
             │  · filters, profiles  │
             │  · live event ingest  │
             └──┬─────────────────┬──┘
                │                 │
       ┌────────▼────────┐  ┌─────▼──────────────┐
       │ Models          │  │ Store (contract)   │  app/db
       │ content         │  │  MongoStore        │
       │ collaborative   │  │  MemoryStore       │
       │ hybrid          │  └─────┬──────────────┘
       │ popularity      │        │
       └────────┬────────┘  ┌─────▼──────────────┐
                │           │ products / users / │
       ┌────────▼────────┐  │ events collections │
       │ ModelBundle     │  └────────────────────┘
       │ (joblib, atomic)│
       └─────────────────┘
```

**Storage abstraction.** Nothing outside `app/db` imports a driver. `Store` is an
abstract base class; `MongoStore` and `MemoryStore` implement it, and the same
parametrised test suite runs against both. Pointing `MONGO_URI` at a local
`mongod` or an Atlas cluster requires no code change.

**Layering.** `app/eval` and `app/training` never import from `app/serving`;
`app/serving` never imports from `app/api`. Dependencies point in one direction,
which is why the pipeline can be exercised headlessly with no HTTP server.

| Module | Responsibility |
|---|---|
| `app/data/synthetic.py` | Seeded clickstream + catalogue generator |
| `app/features/build.py` | Item feature engineering (TF-IDF + taxonomy + numeric) |
| `app/features/interactions.py` | Implicit-feedback matrix, recency decay |
| `app/models/content_based.py` | Profile-centroid recommender |
| `app/models/collaborative.py` | Item-item kNN + Truncated SVD, popularity baseline |
| `app/models/hybrid.py` | RRF and score-blend fusion |
| `app/eval/metrics.py` | Ranking + catalogue-health metrics |
| `app/eval/evaluate.py` | Temporal split, segment scoring |
| `app/eval/tuning.py` | Fusion-weight grid search |
| `app/training/pipeline.py` | ingest → tune → fit → evaluate → persist |
| `app/serving/service.py` | Bundle cache, filtering, live profile updates |

---

### The models

#### `content` — content-based

A user's taste is a **profile vector**: the recency-weighted centroid of the
feature vectors of items they interacted with. Scoring is cosine similarity
between that profile and the catalogue.

Item vectors concatenate three views, each L2-normalised and weighted:

| Block | Encoder | Rationale |
|---|---|---|
| Lexical | TF-IDF word 1–2 grams + char 3–5 grams | tolerant of typos and morphology |
| Taxonomy | one-hot category / subcategory / brand | signal that text alone under-weights |
| Numeric | `log1p` price, Bayesian rating, `log1p` rating count | price affinity is a strong real driver |

Two details that mattered more than expected:

- **`profile_window=10`.** Averaging a user's whole 80-event history flattens
  the profile toward the catalogue mean, which ranks everything nearly equally.
  Restricting the centroid to the strongest recent items keeps it sharp.
- **Mean *and* max pooling.** Mean pooling answers *"which category does this
  user like?"*. Max pooling adds *"which specific thing?"* — it is what
  separates two products inside one subcategory.

Because the profile is an order-free weighted sum, absorbing a new event is a
single vector add. That is what makes the real-time path in
`RecommenderService.record_event` possible without retraining.

#### `collaborative` — item-item kNN + Truncated SVD

Two estimators on the same implicit matrix, min-max normalised per user and
blended with `knn_weight`:

- **Item-item kNN** — cosine similarity between matrix columns ("users who
  bought X also bought Y"), scored as `R @ S`. Damped by **symmetric
  Bell–Koren shrinkage**, `S_ij / (1 + (n_i + n_j)/λ)`. The denominator must be
  symmetric in `(i, j)`: an asymmetric damping silently biases scoring towards
  popular items. (This was a real bug here — see [Testing](#testing).)
- **Truncated SVD** — rank-`f` factorisation; scores are the user factor dotted
  with every item factor. Generalises across sparse items where kNN has no
  co-occurrence evidence at all.

#### `hybrid` — Reciprocal Rank Fusion

Default mode is RRF, `Σ wᵢ / (k + rankᵢ)`, which is scale-free and needs only
the orderings. A linear **score blend** is also available (`mode="score"`).

#### `popularity` — the control

Non-personalised leaderboard. Included deliberately: a personalised model that
cannot beat it is not personalising, it is ranking by date.

#### How the fusion weights are chosen

Grid-searched over 63 combinations on a **validation** slice that sits strictly
between training and test. The test slice is only ever read to produce the final
table.

```
objective = mean NDCG@10 over {all, cold_start} + TUNING_COVERAGE_WEIGHT * mean coverage
```

**The coverage term is a judgement call, and it changes the answer.** With a
pure-accuracy objective the search drives `content_weight` to **0** and the
"hybrid" silently degenerates into plain collaborative filtering. Content-based
ranking contributes little raw accuracy but a great deal of catalogue reach, and
reach is what stops a catalogue collapsing onto the same head items for
everyone. If your objective really is pure NDCG, set
`TUNING_COVERAGE_WEIGHT=0` and accept the degenerate hybrid.

---

### Evaluation

**The split is temporal, per user, and leak-free.** Each user's last `N`
positive events are held out and cut chronologically — the older slice becomes
validation, the newer slice becomes test. Training is additionally truncated at
each user's *own* first held-out event. Without that truncation a user's
trailing `view` events (which occur *after* their last purchase) stay in the
training set, and the model is scored on a period whose tail it has already
seen. That was a real bug in this codebase.

Note the guarantee is **per user, not global**. The split holds out each user's
own latest events, so user A's newest training event can be later than user B's
held-out event. A global time cutoff would drop sparse users entirely.

Two further choices that materially move the numbers:

- **Only positive events are ground truth** (`purchase`, `add_to_cart`,
  `wishlist`). Scoring a `view` as a hit rewards surfacing items users only
  glanced at.
- **Held-out items the user already touched before the cutoff are dropped.** A
  recommender cannot be expected to re-serve something it has seen.

**The served model trains on the full event log; only the evaluation models see
the cut.** Applying the split to the shipped model is a bug with a quiet
symptom: a user whose entire history post-dates their first held-out positive
vanishes from the served matrix and gets cold-start handling despite having a
full history in the store. That happened here (`U00042`, 25 events in the store,
served as cold-start).

**Results are reported per segment** — `cold_start` (≤ 25 training events) and
`warm` (≥ 60), scored from the same fit. Pooling them hides exactly the regime
where the strategies differ, and the segments disagree about which model wins.

**`cold_start: true` in a response means "the trained matrix has no history for
you"**, not "the store has no events for you". A new user who purchases
something still sees `cold_start: true` (the collaborative models have not
retrained) alongside `profile_updated: true`.

Metrics: Precision@K, Recall@K, MAP@K, NDCG@K, MRR, HitRate@K, plus catalogue
**coverage**, **novelty** (−log₂ popularity), **personalisation** (1 − mean
pairwise Jaccard) and **intra-list diversity**.

---

### Results

1,500 users · 800 products · 117,333 events · 63-trial weight grid ·
NDCG@10 on the held-out test slice · model version `b421075e6b77`.
Reproduce with `python -m scripts.train --report`.

| segment | model | NDCG@10 | MAP | Recall | HitRate | Coverage | Novelty | Personalisation |
|---|---|---|---|---|---|---|---|---|
| all | **hybrid** | **0.0999** | **0.0558** | 0.1520 | **0.3485** | 0.819 | 9.22 | 0.983 |
| all | collaborative | 0.0988 | 0.0538 | **0.1536** | 0.3475 | 0.804 | 9.24 | 0.984 |
| all | content | 0.0585 | 0.0298 | 0.0960 | 0.2358 | **0.901** | **9.83** | **0.988** |
| all | popularity | 0.0187 | 0.0096 | 0.0280 | 0.0796 | 0.031 | 8.39 | 0.289 |
| cold_start | **hybrid** | **0.1044** | **0.0502** | **0.1456** | **0.4480** | 0.589 | 9.14 | 0.978 |
| cold_start | collaborative | 0.0977 | 0.0469 | 0.1359 | 0.4160 | 0.578 | 9.17 | 0.980 |
| cold_start | content | 0.0599 | 0.0268 | 0.0909 | 0.3000 | **0.823** | **9.78** | **0.989** |
| cold_start | popularity | 0.0204 | 0.0099 | 0.0255 | 0.1000 | 0.018 | 8.37 | 0.129 |
| warm | **collaborative** | **0.0913** | **0.0517** | **0.1536** | **0.2962** | 0.774 | 9.31 | 0.986 |
| warm | hybrid | 0.0880 | 0.0514 | 0.1456 | 0.2806 | **0.776** | 9.29 | 0.986 |
| warm | content | 0.0561 | 0.0315 | 0.0958 | 0.1938 | 0.803 | 9.87 | 0.987 |
| warm | popularity | 0.0148 | 0.0068 | 0.0275 | 0.0634 | 0.031 | 8.40 | 0.392 |

**Reading these numbers**

1. **The hybrid is the best single choice, and the segments are the only reason
   you can see that.** It wins overall on NDCG (0.0999 vs 0.0988) and MAP, and
   wins cold-start by a clear margin (NDCG 0.1044 vs 0.0977, hit-rate 0.448 vs
   0.416) — while **losing** to collaborative filtering on warm users (0.0880 vs
   0.0913). Content contributes most exactly where interaction data is thin.
   A pooled-only evaluation would have reported a near-tie and hidden this.
2. **Content-based trades accuracy for reach.** 3.1× the popularity baseline on
   NDCG (0.0585 vs 0.0187) but 0.90 coverage against collaborative's 0.80, and
   the highest novelty and personalisation in the table. It is the only strategy
   that can rank a product with zero interactions.
3. **Popularity is the floor, and behaves like one.** 0.0187 NDCG, 0.031
   coverage, personalisation 0.129 cold-start — it serves the same list to
   everyone. Every personalised model beats it by a wide margin, which is the
   evidence that any of them actually personalises.
4. The tuner bought **+3.4% NDCG and +0.09 coverage** over collaborative-only on
   the validation slice (`artifacts/fusion_weights.json`).

**On the absolute values.** NDCG@10 ≈ 0.10 looks low, and that is expected:
each user has 2–3 held-out positives in a catalogue of 800, and a ranker
returning 10 random items scores NDCG@10 ≈ 0.007 in that setting. The hybrid at
0.0999 is roughly **13× random**; the popularity baseline at 0.0187 is about
2.4×. The *relative* ordering between strategies is the transferable result.

---

### API

Base path `/api/v1`. Interactive schema at `/docs`.

#### Recommendations

```http
POST /recommendations
{
  "user_id": "U00042",
  "top_k": 10,
  "strategy": "hybrid",
  "category": "Electronics",
  "brand": "Aurora",
  "min_price": 20.0,
  "max_price": 250.0,
  "tags": ["wireless"],
  "exclude_product_ids": ["P00123"],
  "include_seen": false
}
```

```json
{
  "user_id": "U00042",
  "strategy": "hybrid",
  "top_k": 10,
  "items": [
    {
      "product_id": "P00006",
      "title": "Cascade Dustproof Pantry",
      "category": "Grocery",
      "subcategory": "Pantry",
      "brand": "Cascade",
      "price": 45.14,
      "rating": 3.86,
      "rating_count": 96,
      "tags": ["value-pack", "clearance", "new-arrival", "quiet", "cotton"],
      "popularity": 0.00273,
      "score": 0.014119,
      "rank": 1,
      "reason": "hybrid-rrf"
    }
  ],
  "count": 10,
  "model_version": "b421075e6b77",
  "generated_at": "2026-09-29T15:51:44.902Z",
  "cold_start": false,
  "latency_ms": 7.226
}
```

Abbreviated to the first item; `latency_ms` is one observed sample (7–35 ms
across runs, single-digit after the first call warms the profile cache).

`strategy` ∈ `hybrid` · `collaborative` · `content` · `popularity`.
`GET /users/{user_id}/recommendations` is a query-string convenience wrapper.

#### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/recommendations` | Personalised ranking with optional filters |
| `GET` | `/users/{id}/recommendations` | Same, query-string form |
| `POST` | `/products/{id}/similar` | "More like this", fused across both models |
| `POST` | `/events` | Record an interaction; updates the live profile |
| `GET` | `/users/{id}/profile` | Rolled-up profile, `cold_start` flag |
| `GET` | `/products` | Browse, paginate, filter by category/brand/search |
| `GET` | `/products/facets` | Available categories, brands, tags, price range |
| `GET` | `/products/{id}` | Single product (404 if unknown) |
| `GET` | `/trending` | Global popularity leaderboard |
| `GET` | `/health` | Readiness, backend, counts, model version |
| `GET` | `/models` | Loaded strategies, version, hyperparameters |
| `GET` | `/metrics` | Last offline evaluation |
| `GET` | `/tuning` | Weight-search results |
| `POST` | `/admin/train` | Retrain (background by default) |
| `POST` | `/admin/reload` | Hot-reload the bundle from disk |
| `GET` | `/admin/train/status` | Retrain progress |

#### Real-time feedback loop

```http
POST /events
{"user_id": "U00042", "product_id": "P00123", "event_type": "purchase"}

→ 202 {"event_id": "...", "accepted": true, "profile_updated": true}
```

A `view` costs a write and no model work. A `purchase` / `add_to_cart` /
`wishlist` / `rating` is additionally folded into the user's in-memory content
profile, so **the next recommendation request already reflects it** — no
retrain, no reload. Asserted by
`tests/test_api.py::test_event_changes_the_next_recommendation`.

#### Serving behaviour

- **Cold start never returns an empty list.** Unknown users fall through to the
  popularity component and are served a ranked list with `cold_start: true`.
- **Retraining is non-blocking.** `POST /admin/train` returns immediately and
  keeps serving the old bundle until the new one is fully written.
- **Bundle staleness is bounded** by `MODEL_CACHE_TTL_SECONDS` (60s default), so
  a bundle retrained by another process is picked up without a restart.

---

### Configuration

All settings are environment variables or lines in `.env` (see `.env.example`).

| Variable | Default | Notes |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` | Atlas URIs work unchanged |
| `MONGO_DB` | `reco` | |
| `ALLOW_MEMORY_FALLBACK` | `true` | `false` = fail fast if Mongo is unreachable |
| `MONGO_CONNECT_TIMEOUT_MS` | `1500` | startup probe budget |
| `N_USERS` / `N_PRODUCTS` / `N_EVENTS` | 1500 / 800 / 120000 | synthetic dataset size |
| `RANDOM_SEED` | `42` | dataset is reproducible |
| `SVD_FACTORS` | `64` | latent-factor rank |
| `KNN_TOP_K` | `50` | similarity pruning depth |
| `RECENCY_HALFLIFE_DAYS` | `30` | decay on interaction weights |
| `EVAL_HOLDOUT_PER_USER` | `5` | positives held out per user |
| `TUNING_VAL_SIZE` | `2` | older half of the holdout becomes validation |
| `TUNING_MAX_USERS` | `300` | users sampled per tuning trial |
| `TUNING_COVERAGE_WEIGHT` | `0.05` | reach-vs-accuracy trade-off; `0` = pure NDCG |
| `DEFAULT_TOP_K` / `MAX_TOP_K` | `10` / `100` | response size bounds |
| `MODEL_CACHE_TTL_SECONDS` | `60` | bundle staleness window |

---

### Testing

```bash
python -m pytest -q          # 184 tests, ~50s
ruff check app scripts tests # clean
```

The suite targets defects that produce *plausible but wrong* numbers, not just
line coverage:

- **Metrics are checked against hand-computed values** (`tests/test_metrics.py`).
  A wrong NDCG or MAP silently inverts every model comparison while still
  producing a confident-looking table.
- **The store contract runs against both backends** (`tests/test_store.py`),
  parametrised so identical assertions execute against `MongoStore` and
  `MemoryStore`. MongoDB-dependent tests **skip** if no server is reachable, so
  the suite is green either way — check the skip count before trusting a
  backend-agnostic claim.
- **Ranking invariants hold for every strategy**: contiguous ranks,
  non-increasing scores, never re-serving a seen item, filters and exclusions
  respected, `top_k` clamped to the catalogue, determinism under score ties.
- **The split is proven leak-free** (`tests/test_evaluation.py`): partitions
  disjoint, per-user time ordering, ground truth restricted to positive events,
  served model separated from the evaluation cut.
- **The real-time path is tested end to end**: recording an event must change
  the next response.
- **API tests run against the real app** with a genuinely trained bundle — no
  mocks of the models or the store.

#### Bugs these caught

Recorded because each produced plausible output rather than an error, which is
what makes them worth documenting:

| Bug | How it failed silently | Regression test |
|---|---|---|
| Bell–Koren shrinkage damped per column, not per pair | item-item similarity became **asymmetric**, biasing scores towards popular items | `test_knn_similarity_is_symmetric_before_pruning` |
| `MemoryStore` returned insertion order | profiles and recency weighting disagreed with `MongoStore` | `test_events_are_sorted_for_profiles` |
| Temporal split removed only held-out rows | trailing `view` events stayed in training — **leakage** inflating every metric | `test_held_out_events_are_newer_than_training_per_user` |
| `head(val_size)` over a descending sort | validation was the *newest* events, i.e. part of the test window | `test_val_precedes_test_per_user` |
| Served models fitted on the evaluation split | users with full histories served as cold-start | `test_served_models_use_all_events_not_the_evaluation_cut` |
| `insert_many` sent the whole event log at once | exceeded MongoDB's 16MB BSON limit, aborting the load | `test_bulk_insert_stays_under_the_bson_message_limit` |
| `scripts.ingest` cleared the store before reading it | running it without `--generate` destroyed the data it was about to load | (CLI-level; `clear` is now gated on `generate`) |

`B008` is allowlisted in `pyproject.toml` — it is FastAPI's documented
`Depends()` idiom, not a defect.

---

### Design notes

- **Synchronous pymongo behind async FastAPI.** The models are CPU-bound
  sparse-matrix work with no natural async boundary, so endpoints dispatch to a
  threadpool rather than maintaining a dual sync/async data layer. Reasonable at
  this scale; a service with heavier per-request I/O should use
  `AsyncMongoClient` instead.
- **Artifacts are written to a temp file and atomically swapped.** A crash
  mid-write cannot leave a half-written bundle that fails to load on boot, and
  the previous bundle keeps serving until the new one is complete.
- **Bundles are versioned by a data fingerprint** (SHA-256 of row counts, max
  timestamp, schema), so a stale artifact is identifiable after a retrain.
- **Profile caches are bounded** (`cache_size`, default 20k entries). A
  long-running service sees unbounded user ids and each entry holds an
  n-feature float vector.
- **SVD user factors are cached at fit time.** Re-projecting per user turned a
  300-user evaluation into minutes of work; a known user row is a slice and a
  matmul.
- **`similar` accepts a strategy parameter** so item-to-item ranking is
  consistent with whichever feed the client is using.
- **`recall_at_k` counts distinct items.** A repeated id cannot push recall
  above 1.0 and make a broken ranking look better than a working one.

---

### Honest limitations

**The data is synthetic and the results are not a performance claim.** The
generator has real latent structure and the evaluation protocol is sound, so
the *ranking of strategies* is informative. The absolute NDCG values reflect a
toy catalogue of 800 items with 8 categories and template-generated text. Do not
quote them.

Specific things that are missing or weak:

- **No real-data path.** There is no loader for MovieLens, Amazon reviews or
  your own catalogue. `app/data/` only has the synthetic generator. Wiring in
  real data means writing a loader that emits the same three collections.
- **The weight search is a coarse grid on a 300-user sample.** 63 points, no
  random restarts, no per-segment weights, and it takes ~19 minutes. A proper
  search would use a real optimiser and the full user base.
- **No online or counterfactual evaluation.** Everything here is offline
  next-item prediction. There is no A/B framework, no interleaving, and no
  guard against the feedback loop where recommendations shape the event log
  that trains the next model.
- **No auth, rate limiting or quotas.** Every endpoint is open. `POST /admin/train`
  will happily re-run a 20-minute job on request. This is a research service
  with a production-shaped API, not a deployable service.
- **Live events only update the content model.** `CollaborativeRecommender` is
  immutable after `fit`; new interactions are invisible to it until a retrain.
  There is no incremental factorisation, no online SVD update, and no scheduled
  retrain job — wiring that up is a prerequisite for production.
- **`MemoryStore` is single-process and process-local.** Two workers behind a
  load balancer would each hold a different copy of the data.
- **RRF issues three `recommend()` calls per request.** Single-digit
  milliseconds at this catalogue size, but it wants a batched scoring path
  before the catalogue grows by an order of magnitude.
- **The `min_df=2` TF-IDF default assumes more documents than a tiny catalogue.**
  It is overridden to `1` in the test fixture; a real 200-item catalogue would
  need that too.
- **No item cold-start evaluation.** Content-based's main advantage is ranking
  new products, but no segment isolates items with zero interactions, so the
  README's claim that content is the only strategy that can do this is argued
  from first principles rather than measured.
- **No reproducibility check across environments.** `requirements-lock.txt`
  pins the development environment; the Docker image installs from the loose
  `requirements.txt`.

#### If you take one thing from this

The evaluation protocol is the durable part, not the model numbers. A temporal
per-user split with an explicit cutoff, positive-only ground truth, already-seen
items removed, and a validation slice that sits between train and test — that
transfers to any catalogue. The hybrid-beats-collaborative result is real but
modest (0.0999 vs 0.0988 pooled) and reverses on warm users, so it is a
trade-off decision, not a clean win.

---

### Docker

```bash
docker compose up --build
curl localhost:8000/api/v1/health

docker compose exec api python -m scripts.ingest --generate
docker compose exec api python -m scripts.train
docker compose exec api python -m scripts.smoke_test
```

The compose file sets `ALLOW_MEMORY_FALLBACK=false` so a MongoDB failure is
loud rather than silently degrading to a per-process store.
