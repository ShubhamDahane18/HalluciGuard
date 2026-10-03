"""Benchmark runner — compare all 4 methods on the same dataset."""
from __future__ import annotations
import json
import logging
from typing import Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


def run_all_benchmarks(results_dir: str = "results") -> Dict[str, Any]:
    """Run all four verification methods and return a comparison."""
    from evaluation.evaluator import run_evaluation

    methods = ["embedding", "nli", "llm", "adaptive"]
    all_results = {}
    comparison_rows = []

    for method in methods:
        logger.info(f"Benchmarking method: {method}")
        result = run_evaluation(method=method, results_dir=results_dir)
        all_results[method] = result

        clf = result["classification"]
        eff = result["efficiency"]
        comparison_rows.append({
            "method": method,
            "accuracy": clf["accuracy"],
            "precision": clf["precision"],
            "recall": clf["recall"],
            "f1": clf["f1"],
            "embedding_calls": eff["embedding_calls"],
            "nli_calls": eff["nli_calls"],
            "llm_calls": eff["llm_calls"],
            "llm_avoidance_rate": eff["llm_avoidance_rate"],
            "avg_latency_per_claim": eff["avg_latency_per_claim"],
        })

    df = pd.DataFrame(comparison_rows)
    df.to_csv(f"{results_dir}/comparison.csv", index=False)
    logger.info("Comparison saved to results/comparison.csv")
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON")
    print("=" * 60)
    print(df.to_string(index=False))

    return all_results


if __name__ == "__main__":
    from backend.utils.logging_config import setup_logging
    setup_logging()
    run_all_benchmarks()
