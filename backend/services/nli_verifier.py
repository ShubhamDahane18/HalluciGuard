"""NLI-based claim verifier (Baseline 2)."""
from __future__ import annotations
import logging
from typing import List, Dict, Any

from backend.config import settings

logger = logging.getLogger(__name__)

# ─── Global NLI model cache ──────────────────────────────────────────────
_nli_model = None
_nli_tokenizer = None


def _get_nli_model():
    """Load and cache the NLI model."""
    global _nli_model, _nli_tokenizer
    if _nli_model is not None:
        return _nli_model, _nli_tokenizer

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    model_name = settings.nli_model
    logger.info(f"Loading NLI model: {model_name}")
    try:
        _nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _nli_model.eval()
        logger.info("NLI model loaded successfully")
    except Exception as e:
        logger.warning(
            f"Failed to load {model_name}: {e}. Falling back to cross-encoder/nli-deberta-v3-xsmall"
        )
        fallback = "cross-encoder/nli-deberta-v3-xsmall"
        _nli_tokenizer = AutoTokenizer.from_pretrained(fallback)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(fallback)
        _nli_model.eval()

    return _nli_model, _nli_tokenizer


def _run_nli(premise: str, hypothesis: str) -> Dict[str, float]:
    """
    Run NLI inference for a single premise-hypothesis pair.

    Returns:
        {"entailment": float, "neutral": float, "contradiction": float}
    """
    import torch
    import torch.nn.functional as F

    model, tokenizer = _get_nli_model()

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)[0].tolist()

    # DeBERTa NLI model label order: contradiction, neutral, entailment
    label_names = model.config.id2label
    prob_map = {label_names[i].lower(): p for i, p in enumerate(probs)}

    # Normalise to standard keys
    result = {
        "entailment": prob_map.get("entailment", 0.0),
        "neutral": prob_map.get("neutral", 0.0),
        "contradiction": prob_map.get("contradiction", 0.0),
    }
    return result


def _aggregate_nli(claim: str, evidence_items: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate NLI probabilities across all evidence items.
    Use max entailment and max contradiction strategy.
    """
    if not evidence_items:
        return {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}

    all_probs = []
    for ev in evidence_items[:3]:  # limit to top-3 for speed
        probs = _run_nli(ev["text"], claim)
        all_probs.append(probs)

    # Aggregate: take max entailment, max contradiction, compute neutral
    max_ent = max(p["entailment"] for p in all_probs)
    max_cont = max(p["contradiction"] for p in all_probs)
    avg_neut = sum(p["neutral"] for p in all_probs) / len(all_probs)

    # Renormalize
    total = max_ent + max_cont + avg_neut
    if total == 0:
        total = 1.0
    return {
        "entailment": round(max_ent / total, 4),
        "neutral": round(avg_neut / total, 4),
        "contradiction": round(max_cont / total, 4),
    }


def verify_by_nli(
    claim: str,
    evidence_items: List[Dict[str, Any]],
    confidence_threshold: float | None = None,
) -> Dict[str, Any]:
    """
    Verify a claim against evidence using NLI.

    Maps:
        entailment    → supported
        neutral       → uncertain
        contradiction → unsupported

    Returns:
        {
            "label": str,
            "confidence": float,
            "probabilities": dict,
        }
    """
    confidence_threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else settings.nli_confidence_threshold
    )

    try:
        probs = _aggregate_nli(claim, evidence_items)
    except Exception as e:
        logger.error(f"NLI verification failed: {e}")
        return {
            "label": "uncertain",
            "confidence": 0.0,
            "probabilities": {"entailment": 0.33, "neutral": 0.34, "contradiction": 0.33},
        }

    # Determine dominant label
    dominant = max(probs, key=probs.get)
    confidence = probs[dominant]

    label_map = {
        "entailment": "supported",
        "neutral": "uncertain",
        "contradiction": "unsupported",
    }
    label = label_map.get(dominant, "uncertain")

    logger.debug(f"NLI verification: dominant={dominant}, confidence={confidence:.3f}, label={label}")
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": probs,
    }


def is_nli_model_loaded() -> bool:
    """Check if the NLI model is already loaded."""
    return _nli_model is not None
