"""
Experiment runner for HalluciGuard with MLflow tracking.

Usage:
    python experiments/run_experiment.py --method adaptive
    python experiments/run_experiment.py --method embedding
    python experiments/run_experiment.py --method nli
    python experiments/run_experiment.py --method llm
    python experiments/run_experiment.py --method all
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def run_mlflow_experiment(method: str) -> None:
    import mlflow
    from backend.config import settings
    from evaluation.evaluator import run_evaluation

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    logger.info(f"Starting MLflow experiment for method={method}")

    with mlflow.start_run(run_name=f"{method}_baseline"):
        # Log parameters
        mlflow.log_params({
            "method": method,
            "model": settings.ollama_model,
            "embedding_model": settings.embedding_model,
            "top_k": settings.top_k,
            "similarity_threshold": settings.similarity_threshold,
            "high_similarity_threshold": settings.high_similarity_threshold,
            "low_similarity_threshold": settings.low_similarity_threshold,
            "nli_confidence_threshold": settings.nli_confidence_threshold,
        })

        # Run evaluation
        results = run_evaluation(method=method, results_dir="results")

        clf = results["classification"]
        hall = results["hallucination"]
        eff = results["efficiency"]

        # Log metrics
        mlflow.log_metrics({
            "accuracy": clf["accuracy"],
            "precision": clf["precision"],
            "recall": clf["recall"],
            "f1": clf["f1"],
            "true_hallucination_rate": hall["true_hallucination_rate"],
            "predicted_hallucination_rate": hall["predicted_hallucination_rate"],
            "predicted_faithfulness_score": hall["predicted_faithfulness_score"],
            "embedding_calls": eff["embedding_calls"],
            "nli_calls": eff["nli_calls"],
            "llm_calls": eff["llm_calls"],
            "llm_avoidance_rate": eff["llm_avoidance_rate"],
            "avg_latency_per_claim": eff["avg_latency_per_claim"],
            "total_latency_seconds": eff["total_latency_seconds"],
        })

        # Log artifact (CSV results)
        csv_path = results.get("csv_path")
        if csv_path and os.path.exists(csv_path):
            mlflow.log_artifact(csv_path)

        # Log confusion matrix as JSON artifact
        cm_path = f"results/{method}_confusion_matrix.json"
        with open(cm_path, "w") as f:
            json.dump(clf["confusion_matrix"], f, indent=2)
        mlflow.log_artifact(cm_path)

        logger.info(
            f"Experiment complete — "
            f"accuracy={clf['accuracy']:.3f}, f1={clf['f1']:.3f}, "
            f"llm_avoidance={eff['llm_avoidance_rate']:.1%}"
        )

        # Print summary
        print("\n" + "=" * 60)
        print(f"EXPERIMENT RESULTS — method: {method.upper()}")
        print("=" * 60)
        print(f"  Accuracy:          {clf['accuracy']:.4f}")
        print(f"  Precision:         {clf['precision']:.4f}")
        print(f"  Recall:            {clf['recall']:.4f}")
        print(f"  F1:                {clf['f1']:.4f}")
        print(f"  Faithfulness:      {hall['predicted_faithfulness_score']:.4f}")
        print(f"  Embedding calls:   {eff['embedding_calls']}")
        print(f"  NLI calls:         {eff['nli_calls']}")
        print(f"  LLM calls:         {eff['llm_calls']}")
        print(f"  LLM avoidance:     {eff['llm_avoidance_rate']:.1%}")
        print(f"  Avg latency/claim: {eff['avg_latency_per_claim']:.3f}s")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="HalluciGuard experiment runner")
    parser.add_argument(
        "--method",
        choices=["embedding", "nli", "llm", "adaptive", "all"],
        default="adaptive",
        help="Verification method to evaluate",
    )
    args = parser.parse_args()

    import os
    os.makedirs("results", exist_ok=True)

    if args.method == "all":
        for m in ["embedding", "nli", "llm", "adaptive"]:
            run_mlflow_experiment(m)
    else:
        run_mlflow_experiment(args.method)


if __name__ == "__main__":
    main()
