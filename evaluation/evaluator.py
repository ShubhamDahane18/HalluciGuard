"""Evaluator: run a verification method against the benchmark dataset."""
from __future__ import annotations
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from evaluation.metrics import (
    compute_classification_metrics,
    compute_hallucination_metrics,
    compute_efficiency_metrics,
)

logger = logging.getLogger(__name__)


def load_benchmark(path: str = "data/evaluation/benchmark.json") -> List[Dict[str, Any]]:
    """Load the ground-truth benchmark dataset."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(
    method: str = "adaptive",
    benchmark_path: str = "data/evaluation/benchmark.json",
    results_dir: str = "results",
) -> Dict[str, Any]:
    """
    Run the specified verification method against all benchmark claims.

    Args:
        method: One of "embedding", "nli", "llm", "adaptive"
        benchmark_path: Path to benchmark JSON
        results_dir: Directory to save CSV results

    Returns:
        Dict with classification metrics, hallucination metrics, efficiency metrics
    """
    from backend.utils.logging_config import setup_logging
    from backend.services.retriever import build_index
    from backend.services.adaptive_verifier import verify_single_claim

    setup_logging()
    logger.info(f"Starting evaluation — method={method}")

    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)

    # Ensure index is built
    build_index()

    benchmark = load_benchmark(benchmark_path)

    y_true = []
    y_pred = []
    rows = []

    total_embedding_calls = 0
    total_nli_calls = 0
    total_llm_calls = 0
    total_start = time.time()

    for item in benchmark:
        question = item["question"]
        logger.info(f"Evaluating question: {question!r}")

        for claim_data in item["claims"]:
            claim_text = claim_data["claim"]
            true_label = claim_data["label"]

            # Retrieve evidence
            from backend.services.retriever import retrieve
            evidence = retrieve(claim_text, top_k=5)

            # Verify
            result, counts = verify_single_claim(
                claim_id=0,
                claim_text=claim_text,
                evidence_items=evidence,
                method=method,
            )

            pred_label = result.status
            y_true.append(true_label)
            y_pred.append(pred_label)

            total_embedding_calls += counts.get("embedding", 0)
            total_nli_calls += counts.get("nli", 0)
            total_llm_calls += counts.get("llm", 0)

            rows.append({
                "question": question,
                "claim": claim_text,
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": result.confidence,
                "verification_method": result.verification_method,
                "embedding_score": result.embedding_score,
            })

    elapsed = time.time() - total_start
    total_claims = len(y_true)

    # Compute metrics
    clf_metrics = compute_classification_metrics(y_true, y_pred)
    hall_metrics = compute_hallucination_metrics(y_true, y_pred)
    eff_metrics = compute_efficiency_metrics(
        total_claims=total_claims,
        embedding_calls=total_embedding_calls,
        nli_calls=total_nli_calls,
        llm_calls=total_llm_calls,
        latency_seconds=elapsed,
    )

    # Save CSV
    df = pd.DataFrame(rows)
    csv_path = results_path / f"{method}_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    return {
        "method": method,
        "classification": clf_metrics,
        "hallucination": hall_metrics,
        "efficiency": eff_metrics,
        "csv_path": str(csv_path),
    }
