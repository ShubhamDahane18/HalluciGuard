"""LLM Judge verifier (Baseline 3)."""
from __future__ import annotations
import json
import re
import logging
from typing import List, Dict, Any

from backend.config import settings

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """\
You are a factuality verification system. Your job is to determine whether \
the following factual claim is supported by the provided evidence.

IMPORTANT: Judge ONLY based on the evidence below. Do NOT use outside knowledge.

Claim:
{claim}

Evidence:
{evidence}

Instructions:
- If the evidence clearly supports the claim → label: "supported"
- If the evidence contradicts the claim → label: "unsupported"
- If evidence is insufficient or irrelevant → label: "uncertain"

Return ONLY valid JSON with no markdown, no explanation outside the JSON:
{{
  "label": "supported | uncertain | unsupported",
  "confidence": 0.0,
  "reason": "short explanation in one or two sentences"
}}
"""


def _parse_judge_response(raw: str) -> Dict[str, Any]:
    """Parse LLM judge JSON response with fallback strategies."""
    # Strategy 1: direct parse
    try:
        data = json.loads(raw)
        if "label" in data:
            return data
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract JSON block
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if "label" in data:
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 3: regex field extraction
    label_match = re.search(r'"label"\s*:\s*"(\w+)"', raw)
    conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', raw)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', raw)

    if label_match:
        label = label_match.group(1).lower()
        if label not in ("supported", "uncertain", "unsupported"):
            label = "uncertain"
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        reason = reason_match.group(1) if reason_match else "Unable to determine"
        return {"label": label, "confidence": confidence, "reason": reason}

    return None


def verify_by_llm(
    claim: str,
    evidence_items: List[Dict[str, Any]],
    model: str | None = None,
) -> Dict[str, Any]:
    """
    Verify a claim using an LLM as a judge.

    Returns:
        {
            "used": True,
            "label": str,
            "confidence": float,
            "reason": str,
        }
    """
    from backend.services.answer_generator import generate_with_prompt

    model = model or settings.ollama_model

    if not evidence_items:
        return {
            "used": True,
            "label": "uncertain",
            "confidence": 0.5,
            "reason": "No evidence retrieved to evaluate this claim.",
        }

    # Format top-3 evidence items
    evidence_text = "\n\n".join(
        f"[Source: {e.get('metadata', {}).get('source', 'unknown')}]\n{e['text']}"
        for e in evidence_items[:3]
    )

    prompt = JUDGE_PROMPT.format(claim=claim, evidence=evidence_text)

    # Attempt 1
    try:
        raw = generate_with_prompt(prompt, model=model)
        result = _parse_judge_response(raw)
        if result:
            label = result.get("label", "uncertain").lower()
            if label not in ("supported", "uncertain", "unsupported"):
                label = "uncertain"
            return {
                "used": True,
                "label": label,
                "confidence": float(result.get("confidence", 0.7)),
                "reason": result.get("reason", ""),
            }
    except Exception as e:
        logger.error(f"LLM judge attempt 1 failed: {e}")

    # Attempt 2 (retry)
    try:
        raw = generate_with_prompt(prompt, model=model)
        result = _parse_judge_response(raw)
        if result:
            label = result.get("label", "uncertain").lower()
            if label not in ("supported", "uncertain", "unsupported"):
                label = "uncertain"
            return {
                "used": True,
                "label": label,
                "confidence": float(result.get("confidence", 0.6)),
                "reason": result.get("reason", ""),
            }
    except Exception as e:
        logger.error(f"LLM judge attempt 2 failed: {e}")

    # Safe fallback
    logger.warning("LLM judge returned no parseable result — using safe fallback")
    return {
        "used": True,
        "label": "uncertain",
        "confidence": 0.3,
        "reason": "Verification failed due to LLM parse error.",
    }
