"""Pydantic schemas for HalluciGuard API."""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ─── Request / Response Models ──────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Question to analyze")
    verification_method: str = Field(
        "adaptive",
        description="One of: embedding, nli, llm, adaptive",
    )


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


class VerifyClaimRequest(BaseModel):
    claim: str
    evidence: List[str]
    method: str = "adaptive"


# ─── Intermediate Data Models ─────────────────────────────────────────────


class Claim(BaseModel):
    claim_id: int
    text: str
    type: str = "factual"


class EvidenceItem(BaseModel):
    document_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingResult(BaseModel):
    method: str = "embedding"
    score: float
    label: str  # supported | uncertain | unsupported


class NLIResult(BaseModel):
    label: str  # supported | uncertain | unsupported
    confidence: float
    probabilities: Dict[str, float] = Field(default_factory=dict)


class LLMJudgeResult(BaseModel):
    used: bool = False
    label: str = "uncertain"
    confidence: float = 0.0
    reason: str = ""


class ClaimVerificationResult(BaseModel):
    claim_id: int
    claim: str
    status: str  # supported | uncertain | unsupported
    confidence: float
    verification_method: str
    embedding_score: Optional[float] = None
    nli: Optional[NLIResult] = None
    llm_judge: Optional[LLMJudgeResult] = None
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ─── Summary ──────────────────────────────────────────────────────────────


class HallucinationSummary(BaseModel):
    total_claims: int
    supported: int
    uncertain: int
    unsupported: int
    faithfulness_score: float   # supported / total
    hallucination_rate: float   # unsupported / total
    supported_pct: float
    uncertain_pct: float
    unsupported_pct: float
    embedding_calls: int = 0
    nli_calls: int = 0
    llm_calls: int = 0
    avg_confidence: float = 0.0


class AnalyzeResponse(BaseModel):
    question: str
    answer: str
    model: str
    claims: List[ClaimVerificationResult]
    summary: HallucinationSummary


class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    embedding_model_loaded: bool
    nli_model_loaded: bool
    documents_indexed: int
