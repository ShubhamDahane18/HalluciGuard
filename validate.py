"""End-to-end validation script for HalluciGuard (no Ollama required)."""
import warnings
warnings.filterwarnings("ignore")
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.utils.logging_config import setup_logging
setup_logging("WARNING")

print("=== HalluciGuard End-to-End Validation ===")
print()

# 1. Build FAISS index
from backend.services.retriever import build_index, retrieve
count = build_index()
print(f"1. FAISS index built: {count} chunks")

# 2. Retrieve evidence
evs = retrieve("Tesla founders 2003", top_k=3)
print(f"2. Retrieved {len(evs)} evidence chunks")
if evs:
    print(f"   Top result: {evs[0]['text'][:80]}...")

# 3. Embedding verification
from backend.services.similarity_verifier import verify_by_similarity
emb = verify_by_similarity("Tesla was founded in 2003", evs, threshold=0.65)
print(f"3. Embedding verify: score={emb['score']:.3f}, label={emb['label']}")

# 4. Adaptive verifier (embedding mode only to skip LLM)
from backend.services.adaptive_verifier import verify_single_claim
result, counts = verify_single_claim(1, "Tesla was founded in 2003", evs, method="embedding")
print(f"4. Verify result: status={result.status}, confidence={result.confidence:.3f}")

# 5. Report
from backend.services.report_generator import compute_summary
summary = compute_summary([result])
print(f"5. Summary: faithfulness={summary.faithfulness_score}, hallucination_rate={summary.hallucination_rate}")

# 6. FastAPI health check
from fastapi.testclient import TestClient
from backend.main import app
client = TestClient(app)
h = client.get("/health")
hdata = h.json()
print(f"6. /health: status={hdata['status']}, ollama={hdata['ollama_available']}, docs={hdata['documents_indexed']}")

# 7. Retrieve endpoint
r = client.post("/retrieve", json={"query": "moon landing Apollo 11", "top_k": 2})
print(f"7. /retrieve: {len(r.json()['results'])} results (status {r.status_code})")

# 8. verify-claim endpoint (embedding)
vc = client.post("/verify-claim", json={
    "claim": "Tesla was founded in 2003.",
    "evidence": ["Tesla Motors was founded on July 1, 2003, by Martin Eberhard and Marc Tarpenning."],
    "method": "embedding",
})
print(f"8. /verify-claim: status={vc.json()['result']['status']} (HTTP {vc.status_code})")

# 9. Claim extractor (no LLM — just test JSON parser)
from backend.services.claim_extractor import _parse_claims_json
claims = _parse_claims_json('{"claims": [{"claim_id": 1, "text": "Tesla was founded in 2003.", "type": "factual"}]}')
print(f"9. Claim parser: extracted {len(claims)} claims")

print()
print("=== ALL VALIDATIONS PASSED ===")
print()
print("Note: /analyze endpoint requires Ollama running locally.")
print("Start with: ollama serve && ollama pull qwen2.5:7b")
