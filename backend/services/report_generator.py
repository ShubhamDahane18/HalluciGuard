"""Report generation and hallucination metrics."""
from __future__ import annotations
import logging
from typing import List, Dict, Any

from backend.models.schemas import (
    ClaimVerificationResult,
    HallucinationSummary,
    AnalyzeResponse,
)

logger = logging.getLogger(__name__)


def compute_summary(
    results: List[ClaimVerificationResult],
    call_counts: Dict[str, int] | None = None,
) -> HallucinationSummary:
    """Compute hallucination metrics from claim verification results."""
    total = len(results)
    if total == 0:
        return HallucinationSummary(
            total_claims=0,
            supported=0,
            uncertain=0,
            unsupported=0,
            faithfulness_score=0.0,
            hallucination_rate=0.0,
            supported_pct=0.0,
            uncertain_pct=0.0,
            unsupported_pct=0.0,
        )

    supported = sum(1 for r in results if r.status == "supported")
    uncertain = sum(1 for r in results if r.status == "uncertain")
    unsupported = sum(1 for r in results if r.status == "unsupported")

    call_counts = call_counts or {}

    return HallucinationSummary(
        total_claims=total,
        supported=supported,
        uncertain=uncertain,
        unsupported=unsupported,
        faithfulness_score=round(supported / total, 4),
        hallucination_rate=round(unsupported / total, 4),
        supported_pct=round(supported / total * 100, 2),
        uncertain_pct=round(uncertain / total * 100, 2),
        unsupported_pct=round(unsupported / total * 100, 2),
        embedding_calls=call_counts.get("embedding", 0),
        nli_calls=call_counts.get("nli", 0),
        llm_calls=call_counts.get("llm", 0),
        avg_confidence=round(
            sum(r.confidence for r in results) / total, 4
        ),
    )


def generate_report(
    question: str,
    answer: str,
    model: str,
    results: List[ClaimVerificationResult],
    call_counts: Dict[str, int] | None = None,
) -> AnalyzeResponse:
    """Wrap all results into the full analysis response."""
    summary = compute_summary(results, call_counts)
    logger.info(
        f"Report — total={summary.total_claims}, "
        f"supported={summary.supported}, uncertain={summary.uncertain}, "
        f"unsupported={summary.unsupported}, "
        f"faithfulness={summary.faithfulness_score:.2f}"
    )
    return AnalyzeResponse(
        question=question,
        answer=answer,
        model=model,
        claims=results,
        summary=summary,
    )
