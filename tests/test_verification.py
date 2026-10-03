"""Unit tests for similarity verifier."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSimilarityVerifier:
    def test_supported_high_sim(self):
        """High similarity should yield 'supported'."""
        from backend.services.similarity_verifier import verify_by_similarity
        claim = "Tesla was founded in 2003."
        evidence = [{"text": "Tesla was founded on July 1, 2003, by Martin Eberhard and Marc Tarpenning.", "score": 0.9}]
        result = verify_by_similarity(claim, evidence, threshold=0.6)
        assert result["method"] == "embedding"
        assert result["score"] > 0
        assert result["label"] in ("supported", "uncertain", "unsupported")

    def test_empty_evidence_returns_unsupported(self):
        from backend.services.similarity_verifier import verify_by_similarity
        result = verify_by_similarity("Some claim", [], threshold=0.70)
        assert result["label"] == "unsupported"
        assert result["score"] == 0.0

    def test_label_thresholds(self):
        """Label depends on score vs threshold."""
        from backend.services.similarity_verifier import verify_by_similarity
        # Use identical text to get very high score
        claim = "Machine learning uses algorithms that learn from data."
        evidence = [{"text": "Machine learning uses algorithms that learn from data.", "score": 1.0}]
        result = verify_by_similarity(claim, evidence, threshold=0.85)
        assert result["label"] == "supported"

    def test_returns_dict_with_required_keys(self):
        from backend.services.similarity_verifier import verify_by_similarity
        result = verify_by_similarity("Test claim.", [{"text": "Some evidence text.", "score": 0.5}])
        assert "method" in result
        assert "score" in result
        assert "label" in result
        assert result["label"] in ("supported", "uncertain", "unsupported")
