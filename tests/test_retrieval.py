"""Unit tests for FAISS retriever."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRetriever:
    def test_build_index_returns_positive(self):
        """Building the index should return > 0 chunks."""
        from backend.services.retriever import build_index
        count = build_index()
        assert count > 0, "No documents found — check data/documents/ directory"

    def test_retrieve_returns_list(self):
        from backend.services.retriever import build_index, retrieve
        build_index()
        results = retrieve("Tesla was founded in 2003", top_k=3)
        assert isinstance(results, list)

    def test_retrieve_result_structure(self):
        from backend.services.retriever import build_index, retrieve
        build_index()
        results = retrieve("Elon Musk Tesla CEO", top_k=3)
        for r in results:
            assert "text" in r
            assert "score" in r
            assert "document_id" in r
            assert "metadata" in r

    def test_retrieve_relevance(self):
        """Tesla query should return Tesla document chunks."""
        from backend.services.retriever import build_index, retrieve
        build_index()
        results = retrieve("Tesla founders Martin Eberhard", top_k=5)
        texts = " ".join(r["text"] for r in results).lower()
        assert "tesla" in texts or "eberhard" in texts

    def test_retrieve_top_k_respected(self):
        from backend.services.retriever import build_index, retrieve
        build_index()
        results = retrieve("artificial intelligence", top_k=2)
        assert len(results) <= 2
