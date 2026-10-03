"""FAISS-based evidence retriever with caching."""
from __future__ import annotations
import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from backend.config import settings

logger = logging.getLogger(__name__)

# ─── Global cache (loaded once per process) ──────────────────────────────
_embedding_model = None
_faiss_index = None
_documents: List[Dict[str, Any]] = []  # list of {text, metadata}
_index_path = Path("data/faiss_index")


def _get_embedding_model():
    """Load and cache the sentence-transformers embedding model."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    from sentence_transformers import SentenceTransformer

    model_name = settings.embedding_model
    logger.info(f"Loading embedding model: {model_name}")
    try:
        _embedding_model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully")
    except Exception as e:
        logger.warning(
            f"Failed to load {model_name}: {e}. Falling back to all-MiniLM-L6-v2"
        )
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model


def _embed(texts: List[str]) -> np.ndarray:
    """Embed a list of strings and return numpy array."""
    model = _get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.astype(np.float32)


def _load_documents() -> List[Dict[str, Any]]:
    """Load all text documents from the configured documents directory."""
    docs_path = settings.documents_path
    docs = []
    if not docs_path.exists():
        logger.warning(f"Documents directory not found: {docs_path}")
        return docs

    for fp in sorted(docs_path.glob("*.txt")):
        text = fp.read_text(encoding="utf-8").strip()
        # Chunk by paragraph (~300 tokens each)
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
        for i, para in enumerate(paragraphs):
            docs.append(
                {
                    "text": para,
                    "metadata": {
                        "source": fp.name,
                        "chunk": i,
                        "document_id": f"{fp.stem}_{i:03d}",
                    },
                }
            )
    logger.info(f"Loaded {len(docs)} document chunks from {docs_path}")
    return docs


def build_index(force: bool = False) -> int:
    """
    Build (or reload) the FAISS index from documents.

    Returns:
        Number of indexed document chunks.
    """
    import faiss

    global _faiss_index, _documents

    index_file = _index_path / "index.faiss"
    docs_file = _index_path / "docs.pkl"

    if not force and index_file.exists() and docs_file.exists():
        logger.info("Loading cached FAISS index")
        _faiss_index = faiss.read_index(str(index_file))
        with open(docs_file, "rb") as f:
            _documents = pickle.load(f)
        logger.info(f"Loaded index with {len(_documents)} chunks")
        return len(_documents)

    _documents = _load_documents()
    if not _documents:
        logger.error("No documents found — cannot build index")
        return 0

    texts = [d["text"] for d in _documents]
    embeddings = _embed(texts)
    dim = embeddings.shape[1]

    _faiss_index = faiss.IndexFlatIP(dim)  # Inner product (works with normalized vecs)
    _faiss_index.add(embeddings)

    _index_path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(_faiss_index, str(index_file))
    with open(docs_file, "wb") as f:
        pickle.dump(_documents, f)

    logger.info(f"Built and saved FAISS index with {len(_documents)} chunks")
    return len(_documents)


def retrieve(query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Retrieve the top-K most relevant evidence chunks for a query.

    Returns list of:
        {
            "document_id": str,
            "text": str,
            "score": float,
            "metadata": dict,
        }
    """
    import faiss

    global _faiss_index, _documents

    top_k = top_k or settings.top_k

    if _faiss_index is None or not _documents:
        logger.info("Index not loaded — building now")
        build_index()

    if _faiss_index is None or not _documents:
        logger.error("Cannot retrieve: no index available")
        return []

    q_emb = _embed([query])
    scores, indices = _faiss_index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_documents):
            continue
        doc = _documents[idx]
        results.append(
            {
                "document_id": doc["metadata"]["document_id"],
                "text": doc["text"],
                "score": float(score),
                "metadata": doc["metadata"],
            }
        )
    return results


def get_document_count() -> int:
    """Return number of indexed document chunks."""
    if _documents:
        return len(_documents)
    try:
        docs_file = _index_path / "docs.pkl"
        if docs_file.exists():
            with open(docs_file, "rb") as f:
                docs = pickle.load(f)
            return len(docs)
    except Exception:
        pass
    return 0
