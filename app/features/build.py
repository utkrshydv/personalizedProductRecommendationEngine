"""Item feature engineering.

Every product is turned into one dense vector combining three signals:

* lexical  - TF-IDF (word 1-2 grams + char 3-5 grams) over title, description
             and tags;
* taxonomy - one-hot category / subcategory / brand;
* numeric  - log price, average rating and log rating count, all scaled.

The blocks are concatenated with weights and L2-normalised, so the resulting
cosine similarity is a weighted blend of the three views of a product.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TAG_RE = re.compile(r"[^a-z0-9]+")
TOKEN_RE = re.compile(r"[a-z0-9]+")

TEXT_COLUMNS = ("title", "description", "tags")
TAXONOMY_COLUMNS = ("category", "subcategory", "brand")
NUMERIC_COLUMNS = ("log_price", "rating", "log_rating_count")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        value = " ".join(str(v) for v in value)
    return " ".join(TOKEN_RE.findall(TAG_RE.sub(" ", str(value).lower())))


def build_text_series(products: pd.DataFrame) -> pd.Series:
    """title and description get repeated so they dominate the TF-IDF signal."""
    parts = []
    for _, row in products.iterrows():
        tags = " ".join(row.get("tags", []) or [])
        parts.append(
            " ".join(
                [
                    clean_text(row.get("title", "")) * 2,
                    clean_text(row.get("category", "")),
                    clean_text(row.get("subcategory", "")),
                    clean_text(row.get("brand", "")),
                    clean_text(tags) * 2,
                    clean_text(row.get("description", "")),
                ]
            )
        )
    return pd.Series(parts, index=products.index, dtype="string").fillna("")


def add_numeric_columns(products: pd.DataFrame) -> pd.DataFrame:
    out = products.copy()
    price = pd.to_numeric(out.get("price"), errors="coerce").clip(lower=0.01)
    out["log_price"] = np.log1p(price)
    count = pd.to_numeric(out.get("rating_count"), errors="coerce").fillna(0.0).clip(lower=0)
    out["log_rating_count"] = np.log1p(count)
    out["rating"] = pd.to_numeric(out.get("rating"), errors="coerce").fillna(3.5)
    return out


@dataclass
class ItemFeatureBuilder:
    """Fit once at training time, reused for every scoring call."""

    word_weight: float = 1.0
    char_weight: float = 0.4
    taxonomy_weight: float = 0.6
    numeric_weight: float = 0.25
    min_df: int = 2

    word_vectorizer: TfidfVectorizer | None = field(default=None, init=False)
    char_vectorizer: TfidfVectorizer | None = field(default=None, init=False)
    taxonomy_encoder: OneHotEncoder | None = field(default=None, init=False)
    scaler: StandardScaler | None = field(default=None, init=False)
    product_ids: list[str] = field(default_factory=list, init=False)
    index_by_id: dict[str, int] = field(default_factory=dict, init=False)
    _products: pd.DataFrame = field(default_factory=pd.DataFrame, init=False, repr=False)

    # -- fitting ---------------------------------------------------------
    def fit(self, products: pd.DataFrame) -> ItemFeatureBuilder:
        products = add_numeric_columns(products)
        text = build_text_series(products)

        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=self.min_df,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=max(self.min_df, 3),
            sublinear_tf=True,
        )
        self.word_vectorizer.fit(text.tolist())
        self.char_vectorizer.fit(text.tolist())

        self.taxonomy_encoder = OneHotEncoder(
            handle_unknown="ignore", min_frequency=1, sparse_output=True
        )
        self.taxonomy_encoder.fit(
            products[list(TAXONOMY_COLUMNS)].astype("string").fillna("").to_numpy()
        )

        self.scaler = StandardScaler()
        self.scaler.fit(products[list(NUMERIC_COLUMNS)].to_numpy(dtype=float))

        self.product_ids = products["product_id"].astype(str).tolist()
        self.index_by_id = {pid: i for i, pid in enumerate(self.product_ids)}
        self._products = products.reset_index(drop=True)
        return self

    # -- transform -------------------------------------------------------
    def transform(self, products: pd.DataFrame) -> sparse.csr_matrix:
        if self.word_vectorizer is None or self.char_vectorizer is None:
            raise RuntimeError("ItemFeatureBuilder must be fitted before transform()")
        products = add_numeric_columns(products)
        text = build_text_series(products).tolist()

        word = self.word_vectorizer.transform(text) * self.word_weight
        char = self.char_vectorizer.transform(text) * self.char_weight
        taxonomy = self.taxonomy_encoder.transform(
            products[list(TAXONOMY_COLUMNS)].astype("string").fillna("").to_numpy()
        ) * self.taxonomy_weight
        numeric = sparse.csr_matrix(
            np.abs(self.scaler.transform(products[list(NUMERIC_COLUMNS)].to_numpy(float)))
        ) * self.numeric_weight

        return l2_normalize(sparse.hstack([word, char, taxonomy, numeric]).tocsr())

    def fit_transform(self, products: pd.DataFrame) -> sparse.csr_matrix:
        return self.fit(products).transform(products)

    # -- convenience -----------------------------------------------------
    def vector_for(self, product_id: str) -> sparse.csr_matrix:
        """Feature vector of a single already-fitted product."""
        if product_id not in self.index_by_id:
            raise KeyError(product_id)
        row = self._products.iloc[[self.index_by_id[product_id]]]
        return self.transform(row)

    def transform_subset(self, product_ids: list[str]) -> sparse.csr_matrix:
        """Feature rows for an arbitrary subset of fitted products."""
        missing = [p for p in product_ids if p not in self.index_by_id]
        if missing:
            raise KeyError(missing[0])
        rows = [self.index_by_id[p] for p in product_ids]
        return self.transform(self._products.iloc[rows])

    @property
    def n_features(self) -> int:
        if self.word_vectorizer is None or self.char_vectorizer is None:
            return 0
        return len(self.word_vectorizer.vocabulary_) + len(self.char_vectorizer.vocabulary_)

    @property
    def n_items(self) -> int:
        return len(self.product_ids)


def l2_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Row-wise L2 normalisation, safe for all-zero rows."""
    from sklearn.preprocessing import normalize

    return normalize(matrix, norm="l2", axis=1, copy=False)


def to_dense_rows(matrix: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(matrix.todense())


def cosine_similarity_topk(
    query: sparse.csr_matrix, item_matrix: sparse.csr_matrix, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity of every query row against the catalogue."""
    from sklearn.metrics.pairwise import cosine_similarity

    sims = cosine_similarity(query, item_matrix)
    k = min(top_k, sims.shape[1])
    idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(sims.shape[0])[:, None]
    order = np.argsort(-sims[rows, idx], axis=1)
    idx = idx[rows, order]
    scores = sims[rows, idx]
    return idx, scores


__all__ = [
    "NUMERIC_COLUMNS",
    "TAXONOMY_COLUMNS",
    "ItemFeatureBuilder",
    "add_numeric_columns",
    "build_text_series",
    "clean_text",
    "cosine_similarity_topk",
    "l2_normalize",
    "to_dense_rows",
]
