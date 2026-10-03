"""API endpoint tests using FastAPI TestClient."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "ollama_available" in data
        assert "documents_indexed" in data
        assert data["status"] == "ok"


class TestRetrieveEndpoint:
    def test_retrieve_endpoint(self):
        response = client.post("/retrieve", json={"query": "Tesla founders", "top_k": 3})
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_retrieve_returns_documents(self):
        response = client.post("/retrieve", json={"query": "Elon Musk CEO", "top_k": 5})
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) > 0

    def test_retrieve_result_structure(self):
        response = client.post("/retrieve", json={"query": "NASA moon landing", "top_k": 2})
        data = response.json()
        for r in data["results"]:
            assert "text" in r
            assert "score" in r


class TestVerifyClaimEndpoint:
    def test_verify_claim_embedding(self):
        response = client.post("/verify-claim", json={
            "claim": "Tesla was founded in 2003.",
            "evidence": ["Tesla Motors was founded on July 1, 2003, by Martin Eberhard and Marc Tarpenning."],
            "method": "embedding",
        })
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "call_counts" in data
        assert data["result"]["status"] in ("supported", "uncertain", "unsupported")

    def test_verify_claim_bad_method(self):
        response = client.post("/verify-claim", json={
            "claim": "Some claim",
            "evidence": ["Some evidence"],
            "method": "invalid_method",
        })
        # Should not crash — falls through to adaptive
        assert response.status_code in (200, 422)


class TestAnalyzeEndpoint:
    def test_analyze_empty_question_fails(self):
        """Empty/whitespace questions should be rejected (400 from route or 422 from Pydantic min_length)."""
        response = client.post("/analyze", json={"question": "  "})
        assert response.status_code in (400, 422), f"Expected 400 or 422, got {response.status_code}"

    def test_analyze_invalid_method_fails(self):
        response = client.post("/analyze", json={
            "question": "Who founded Tesla?",
            "verification_method": "invalid",
        })
        assert response.status_code == 400
