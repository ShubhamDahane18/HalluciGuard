"""Evaluation metrics for HalluciGuard experiments."""
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


LABEL_MAP = {"supported": 0, "uncertain": 1, "unsupported": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def encode_labels(labels: List[str]) -> List[int]:
    """Convert string labels to integers."""
    return [LABEL_MAP.get(l, 1) for l in labels]


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, Any]:
    """
    Compute classification metrics.

    Returns:
        {
            "accuracy": float,
            "precision": float,
            "recall": float,
            "f1": float,
            "confusion_matrix": list,
            "per_class": dict,
        }
    """
    y_true_enc = encode_labels(y_true)
    y_pred_enc = encode_labels(y_pred)

    acc = accuracy_score(y_true_enc, y_pred_enc)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true_enc, y_pred_enc, average="weighted", zero_division=0
    )

    # Per-class
    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
        y_true_enc, y_pred_enc, labels=[0, 1, 2], zero_division=0
    )
    per_class = {}
    for i, label in INV_LABEL_MAP.items():
        per_class[label] = {
            "precision": round(float(prec_c[i]), 4),
            "recall": round(float(rec_c[i]), 4),
            "f1": round(float(f1_c[i]), 4),
            "support": int(sup_c[i]),
        }

    cm = confusion_matrix(y_true_enc, y_pred_enc, labels=[0, 1, 2]).tolist()

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm,
        "per_class": per_class,
    }


def compute_hallucination_metrics(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, float]:
    """Project-specific hallucination metrics."""
    total = len(y_true)
    if total == 0:
        return {}

    true_unsupported = sum(1 for l in y_true if l == "unsupported")
    pred_unsupported = sum(1 for l in y_pred if l == "unsupported")
    pred_supported = sum(1 for l in y_pred if l == "supported")

    true_hallucination_rate = true_unsupported / total
    pred_hallucination_rate = pred_unsupported / total
    pred_faithfulness = pred_supported / total

    return {
        "true_hallucination_rate": round(true_hallucination_rate, 4),
        "predicted_hallucination_rate": round(pred_hallucination_rate, 4),
        "predicted_faithfulness_score": round(pred_faithfulness, 4),
    }


def compute_efficiency_metrics(
    total_claims: int,
    embedding_calls: int,
    nli_calls: int,
    llm_calls: int,
    latency_seconds: float,
) -> Dict[str, Any]:
    """System efficiency metrics."""
    llm_avoidance = 1.0 - (llm_calls / total_claims) if total_claims > 0 else 0.0
    avg_latency = latency_seconds / total_claims if total_claims > 0 else 0.0

    return {
        "total_claims": total_claims,
        "embedding_calls": embedding_calls,
        "nli_calls": nli_calls,
        "llm_calls": llm_calls,
        "llm_avoidance_rate": round(llm_avoidance, 4),
        "total_latency_seconds": round(latency_seconds, 2),
        "avg_latency_per_claim": round(avg_latency, 3),
    }
