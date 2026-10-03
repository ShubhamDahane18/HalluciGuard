"""Adaptive cascading verification strategy (Proposed Method)."""
from __future__ import annotations
import logging
from typing import List, Dict, Any

from backend.config import settings
from backend.models.schemas import (
    ClaimVerificationResult,
    EmbeddingResult,
    NLIResult,
    LLMJudgeResult,
    EvidenceItem,
)

logger = logging.getLogger(__name__)


def _make_evidence_items(raw_evidence: List[Dict[str, Any]]) -> List[EvidenceItem]:
    return [
        EvidenceItem(
            document_id=e.get("document_id", "unknown"),
            text=e["text"],
            score=e.get("score", 0.0),
            metadata=e.get("metadata", {}),
        )
        for e in raw_evidence
    ]


def verify_single_claim(
    claim_id: int,
    claim_text: str,
    evidence_items: List[Dict[str, Any]],
    method: str = "adaptive",
) -> tuple[ClaimVerificationResult, Dict[str, int]]:
    """
    Verify a single claim using the specified method.

    Returns:
        (ClaimVerificationResult, call_counts)
        where call_counts = {"embedding": int, "nli": int, "llm": int}
    """
    from backend.services.similarity_verifier import verify_by_similarity
    from backend.services.nli_verifier import verify_by_nli
    from backend.services.llm_verifier import verify_by_llm

    call_counts = {"embedding": 0, "nli": 0, "llm": 0}
    ev_items = _make_evidence_items(evidence_items)

    # ─ Embedding Baseline ───────────────────────────────────────────────
    if method == "embedding":
        call_counts["embedding"] = 1
        emb = verify_by_similarity(claim_text, evidence_items)
        return (
            ClaimVerificationResult(
                claim_id=claim_id,
                claim=claim_text,
                status=emb["label"],
                confidence=emb["score"],
                verification_method="embedding",
                embedding_score=emb["score"],
                evidence=ev_items,
            ),
            call_counts,
        )

    # ─ NLI Baseline ─────────────────────────────────────────────────────
    if method == "nli":
        call_counts["nli"] = 1
        nli = verify_by_nli(claim_text, evidence_items)
        return (
            ClaimVerificationResult(
                claim_id=claim_id,
                claim=claim_text,
                status=nli["label"],
                confidence=nli["confidence"],
                verification_method="nli",
                nli=NLIResult(
                    label=nli["label"],
                    confidence=nli["confidence"],
                    probabilities=nli["probabilities"],
                ),
                evidence=ev_items,
            ),
            call_counts,
        )

    # ─ LLM Baseline ─────────────────────────────────────────────────────
    if method == "llm":
        call_counts["llm"] = 1
        llm = verify_by_llm(claim_text, evidence_items)
        return (
            ClaimVerificationResult(
                claim_id=claim_id,
                claim=claim_text,
                status=llm["label"],
                confidence=llm["confidence"],
                verification_method="llm",
                llm_judge=LLMJudgeResult(
                    used=True,
                    label=llm["label"],
                    confidence=llm["confidence"],
                    reason=llm["reason"],
                ),
                evidence=ev_items,
            ),
            call_counts,
        )

    # ─ Adaptive Cascading Strategy ───────────────────────────────────────
    # Step 1: Embedding
    call_counts["embedding"] = 1
    emb = verify_by_similarity(claim_text, evidence_items)
    emb_score = emb["score"]

    high_thresh = settings.high_similarity_threshold  # default 0.85
    low_thresh = settings.low_similarity_threshold  # default 0.40

    if emb_score >= high_thresh:
        # High confidence from embedding — no need for NLI or LLM
        logger.debug(
            f"Claim {claim_id} resolved by embedding (score={emb_score:.3f})"
        )
        return (
            ClaimVerificationResult(
                claim_id=claim_id,
                claim=claim_text,
                status="supported",
                confidence=emb_score,
                verification_method="adaptive(embedding)",
                embedding_score=emb_score,
                evidence=ev_items,
            ),
            call_counts,
        )

    # Step 2: NLI for uncertain / potentially-unsupported claims
    call_counts["nli"] = 1
    nli = verify_by_nli(claim_text, evidence_items)
    nli_confidence = nli["confidence"]
    nli_label = nli["label"]

    nli_thresh = settings.nli_confidence_threshold  # default 0.80

    if nli_confidence >= nli_thresh:
        # NLI is confident enough
        logger.debug(
            f"Claim {claim_id} resolved by NLI (label={nli_label}, conf={nli_confidence:.3f})"
        )
        return (
            ClaimVerificationResult(
                claim_id=claim_id,
                claim=claim_text,
                status=nli_label,
                confidence=nli_confidence,
                verification_method="adaptive(nli)",
                embedding_score=emb_score,
                nli=NLIResult(
                    label=nli_label,
                    confidence=nli_confidence,
                    probabilities=nli["probabilities"],
                ),
                evidence=ev_items,
            ),
            call_counts,
        )

    # Step 3: LLM Judge for difficult cases
    call_counts["llm"] = 1
    logger.debug(f"Claim {claim_id} escalating to LLM judge")
    llm = verify_by_llm(claim_text, evidence_items)

    return (
        ClaimVerificationResult(
            claim_id=claim_id,
            claim=claim_text,
            status=llm["label"],
            confidence=llm["confidence"],
            verification_method="adaptive(llm)",
            embedding_score=emb_score,
            nli=NLIResult(
                label=nli_label,
                confidence=nli_confidence,
                probabilities=nli["probabilities"],
            ),
            llm_judge=LLMJudgeResult(
                used=True,
                label=llm["label"],
                confidence=llm["confidence"],
                reason=llm["reason"],
            ),
            evidence=ev_items,
        ),
        call_counts,
    )


def verify_all_claims(
    claims: List[Dict[str, Any]],
    method: str = "adaptive",
    top_k: int | None = None,
) -> tuple[List[ClaimVerificationResult], Dict[str, int]]:
    """
    Verify all claims. Retrieves evidence for each claim automatically.

    Returns:
        (results, total_call_counts)
    """
    from backend.services.retriever import retrieve

    top_k = top_k or settings.top_k
    results = []
    total_counts = {"embedding": 0, "nli": 0, "llm": 0}

    for claim in claims:
        claim_id = claim.get("claim_id", 0)
        claim_text = claim.get("text", "")
        logger.info(f"Verifying claim {claim_id}: {claim_text[:60]}...")

        evidence = retrieve(claim_text, top_k=top_k)
        result, counts = verify_single_claim(claim_id, claim_text, evidence, method=method)
        results.append(result)

        for k in total_counts:
            total_counts[k] += counts.get(k, 0)

    return results, total_counts
