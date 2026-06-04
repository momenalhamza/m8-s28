"""Module 8 — Thursday Stretch (Honors Track): Cross-Encoder Re-Ranking.

Add a cross-encoder re-ranking stage to the lab's hybrid retriever and
evaluate the cost/benefit. Cross-encoders score (query, passage) pairs
jointly rather than independently — they produce a more discriminative
ranking, but at a real latency cost.

Use cross-encoder/ms-marco-MiniLM-L-6-v2 from sentence-transformers.
"""

from __future__ import annotations

import numpy as np
import weaviate
from sentence_transformers import CrossEncoder

from retrieval_helpers import CLASS_NAME, hybrid_search

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Module-level singleton — loaded once, reused across all calls.
_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def cross_encoder_rerank(query: str, candidates: list[dict], k_out: int = 5) -> list[str]:
    """Re-rank a candidate list using a cross-encoder.

    `candidates` is a list of {"doc_id": str, "text": str} (or a similar
    schema providing the text to score). Score each (query, candidate.text)
    pair; sort descending; return the top-`k_out` doc_id strings.
    """
    if not candidates:
        return []

    ce = _get_cross_encoder()
    pairs = [(query, c["text"]) for c in candidates]
    scores = ce.predict(pairs)

    # argsort descending, take top k_out
    ranked_indices = np.argsort(scores)[::-1][:k_out]
    return [candidates[i]["doc_id"] for i in ranked_indices]


def _hybrid_search_with_text(
    client: weaviate.Client,
    query: str,
    k: int,
    embedder,
    alpha: float = 0.5,
) -> list[dict]:
    """Hybrid search returning both doc_id and text in one Weaviate round-trip."""
    qv = embedder.encode(query, convert_to_numpy=True).tolist()
    res = (
        client.query.get(CLASS_NAME, ["doc_id", "text"])
        .with_hybrid(query=query, vector=qv, alpha=alpha)
        .with_limit(k)
        .do()
    )
    items = res.get("data", {}).get("Get", {}).get(CLASS_NAME, []) or []
    return [{"doc_id": it["doc_id"], "text": it["text"]} for it in items]


def rerank_search(
    client: weaviate.Client,
    query: str,
    embedder,
    k_in: int = 50,
    k_out: int = 5,
) -> list[str]:
    """Two-stage retriever: hybrid retrieve k_in, cross-encoder re-rank to k_out.

    Stage 1: hybrid_search(client, query, k_in, embedder, alpha=0.5) -> list[doc_id]
    Stage 2: resolve each doc_id back to its text from Weaviate
    Stage 3: cross_encoder_rerank(query, candidates, k_out)

    Return the ordered list of doc_id strings, length <= k_out.
    """
    # Stage 1: hybrid retrieval (text fetched in same round-trip)
    candidates = _hybrid_search_with_text(client, query, k_in, embedder, alpha=0.5)
    if not candidates:
        return []

    # Stage 2: cross-encoder re-ranking
    return cross_encoder_rerank(query, candidates, k_out)
