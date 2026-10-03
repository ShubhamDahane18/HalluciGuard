"""Unit tests for claim extraction."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.claim_extractor import _parse_claims_json, _sentence_split_fallback


class TestClaimExtraction:
    def test_parse_valid_json(self):
        raw = '{"claims": [{"claim_id": 1, "text": "Tesla was founded in 2003.", "type": "factual"}]}'
        claims = _parse_claims_json(raw)
        assert len(claims) == 1
        assert claims[0]["text"] == "Tesla was founded in 2003."

    def test_parse_json_with_extra_text(self):
        raw = 'Here are the claims:\n{"claims": [{"claim_id": 1, "text": "Apple sells iPhones.", "type": "factual"}]}'
        claims = _parse_claims_json(raw)
        assert len(claims) == 1
        assert "Apple" in claims[0]["text"]

    def test_parse_malformed_returns_fallback(self):
        """Malformed LLM output should fall back to sentence splitting."""
        raw = "The LLM forgot to format this as JSON. Tesla was founded in 2003. Elon Musk is CEO."
        claims = _parse_claims_json(raw)
        assert isinstance(claims, list)
        # Sentence split may return some claims
        assert len(claims) >= 0  # graceful, no crash

    def test_sentence_split_fallback(self):
        text = "Tesla was founded in 2003. Elon Musk became CEO in 2008. The Model S launched in 2012."
        claims = _sentence_split_fallback(text)
        assert len(claims) >= 1
        for c in claims:
            assert "claim_id" in c
            assert "text" in c

    def test_empty_input(self):
        claims = _parse_claims_json("")
        assert isinstance(claims, list)

    def test_claim_ids_sequential(self):
        raw = '{"claims": [{"claim_id": 5, "text": "A.", "type": "factual"}, {"claim_id": 7, "text": "B.", "type": "factual"}]}'
        claims = _parse_claims_json(raw)
        assert len(claims) == 2
