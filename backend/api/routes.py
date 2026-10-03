"""FastAPI routes for HalluciGuard."""
from __future__ import annotations
import logging
import time
from typing import List

from fastapi import APIRouter, HTTPException, Body

from backend.config import settings
from backend.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    RetrieveRequest,
    VerifyClaimRequest,
    HealthResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    from backend.services.answer_generator import _check_ollama_available
    from backend.services.retriever import get_document_count, _embedding_model
    from backend.services.nli_verifier import is_nli_model_loaded

    ollama_ok = _check_ollama_available()
    doc_count = get_document_count()
    emb_loaded = _embedding_model is not None

    return HealthResponse(
        status="ok",
        ollama_available=ollama_ok,
        embedding_model_loaded=emb_loaded,
        nli_model_loaded=is_nli_model_loaded(),
        documents_indexed=doc_count,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    """
    Full pipeline: generate answer → extract claims → retrieve evidence →
    verify claims → generate report.
    """
    from backend.services.answer_generator import generate_answer
    from backend.services.claim_extractor import extract_claims
    from backend.services.adaptive_verifier import verify_all_claims
    from backend.services.report_generator import generate_report
    from backend.services.retriever import build_index

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    method = request.verification_method
    if method not in ("embedding", "nli", "llm", "adaptive"):
        raise HTTPException(
            status_code=400,
            detail="verification_method must be one of: embedding, nli, llm, adaptive",
        )

    start = time.time()
    logger.info(f"Analyzing question: {request.question!r} [method={method}]")

    # Ensure index is built
    doc_count = build_index()
    if doc_count == 0:
        raise HTTPException(
            status_code=503,
            detail=f"No documents indexed. Add .txt files to '{settings.documents_dir}/'",
        )

    # Step 1: Generate answer
    try:
        gen_result = generate_answer(request.question)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    answer = gen_result["answer"]
    if not answer:
        raise HTTPException(status_code=500, detail="LLM returned an empty answer.")

    logger.info(f"Answer generated ({len(answer)} chars)")

    # Step 2: Extract claims
    claims = extract_claims(answer)
    if not claims:
        raise HTTPException(status_code=500, detail="No claims could be extracted from the answer.")

    logger.info(f"Extracted {len(claims)} claims")

    # Step 3+4: Retrieve + Verify
    results, call_counts = verify_all_claims(claims, method=method)

    elapsed = round(time.time() - start, 2)
    logger.info(f"Analysis complete in {elapsed}s")

    return generate_report(
        question=request.question,
        answer=answer,
        model=gen_result["model"],
        results=results,
        call_counts=call_counts,
    )


@router.post("/retrieve")
def retrieve_evidence(request: RetrieveRequest):
    """Retrieve evidence for a query."""
    from backend.services.retriever import retrieve, build_index

    build_index()
    results = retrieve(request.query, top_k=request.top_k)
    return {"query": request.query, "results": results}


@router.post("/verify-claim")
def verify_claim(request: VerifyClaimRequest):
    """Verify a single claim against provided evidence texts."""
    from backend.services.adaptive_verifier import verify_single_claim

    evidence_items = [{"text": e, "document_id": f"provided_{i}", "score": 1.0, "metadata": {}} for i, e in enumerate(request.evidence)]
    result, counts = verify_single_claim(
        claim_id=1,
        claim_text=request.claim,
        evidence_items=evidence_items,
        method=request.method,
    )
    return {"result": result.model_dump(), "call_counts": counts}
