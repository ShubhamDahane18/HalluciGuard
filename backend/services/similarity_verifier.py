"""Embedding-based cosine similarity verifier (Baseline 1)."""
from __future__ import annotations
import logging
from typing import List, Dict, Any

import numpy as np

from backend.config import settings

logger = logging.getLogger(__name__)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def verify_by_similarity(
    claim: str,
    evidence_items: List[Dict[str, Any]],
    threshold: float | None = None,
) -> Dict[str, Any]:
    """
    Verify a claim against evidence using embedding cosine similarity.

    The score is the max similarity across all evidence items.
    Label thresholds:
        score >= threshold         → supported
        score >= 0.40              → uncertain
        score < 0.40               → unsupported

    Returns:
        {
            "method": "embedding",
            "score": float,
            "label": "supported" | "uncertain" | "unsupported",
        }
    """
    from backend.services.retriever import _embed  # use shared model

    threshold = threshold if threshold is not None else settings.similarity_threshold

    if not evidence_items:
        return {"method": "embedding", "score": 0.0, "label": "unsupported"}

    claim_emb = _embed([claim])[0]
    evidence_texts = [e["text"] for e in evidence_items]
    ev_embs = _embed(evidence_texts)

    sims = [_cosine_sim(claim_emb, ev_emb) for ev_emb in ev_embs]
    max_sim = max(sims)

    if max_sim >= threshold:
        label = "supported"
    elif max_sim >= 0.40:
        label = "uncertain"
    else:
        label = "unsupported"

    logger.debug(f"Embedding verification: score={max_sim:.3f}, label={label}")
    return {"method": "embedding", "score": round(max_sim, 4), "label": label}
