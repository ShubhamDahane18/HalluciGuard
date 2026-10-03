"""Claim extraction from LLM-generated answers."""
from __future__ import annotations
import json
import re
import logging
from typing import List, Dict, Any

from backend.config import settings
from backend.services.answer_generator import generate_with_prompt

logger = logging.getLogger(__name__)


EXTRACTION_PROMPT = """\
You are a factual claim extractor. Given a text, extract all distinct, atomic, \
verifiable factual claims.

Rules:
- Each claim must be a single, self-contained factual statement.
- Do NOT include opinions, questions, or hedged statements.
- Keep claims concise. Include relevant subject names.
- Return ONLY valid JSON — no markdown, no explanation.

Text:
{text}

Return ONLY this JSON format:
{{
  "claims": [
    {{"claim_id": 1, "text": "...", "type": "factual"}},
    {{"claim_id": 2, "text": "...", "type": "factual"}}
  ]
}}
"""


def _parse_claims_json(raw: str) -> List[Dict[str, Any]]:
    """Parse claims JSON from LLM output with multiple fallback strategies."""
    # Strategy 1: direct parse
    try:
        data = json.loads(raw)
        if "claims" in data:
            return data["claims"]
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract JSON block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if "claims" in data:
                return data["claims"]
        except json.JSONDecodeError:
            pass

    # Strategy 3: extract array of objects
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group())
            if items and isinstance(items[0], dict):
                return items
        except json.JSONDecodeError:
            pass

    # Strategy 4: sentence splitting fallback
    logger.warning("LLM claim extraction failed — falling back to sentence splitting")
    return _sentence_split_fallback(raw)


def _sentence_split_fallback(text: str) -> List[Dict[str, Any]]:
    """Simple sentence splitting as a last-resort claim extractor."""
    # Remove common artifacts from malformed JSON attempts
    cleaned = re.sub(r'[{}\[\]"\'\\]', " ", text)
    # Split on periods, exclamation marks, and newlines
    sentences = re.split(r"[.!?\n]+", cleaned)
    claims = []
    cid = 1
    for s in sentences:
        s = s.strip()
        if len(s) > 15:  # filter very short fragments
            claims.append({"claim_id": cid, "text": s, "type": "factual"})
            cid += 1
    return claims


def extract_claims(text: str) -> List[Dict[str, Any]]:
    """
    Extract atomic factual claims from generated answer text.

    Returns a list of claim dicts:
        [{"claim_id": 1, "text": "...", "type": "factual"}, ...]
    """
    logger.info(f"Extracting claims from text ({len(text)} chars)")

    prompt = EXTRACTION_PROMPT.format(text=text)

    # First attempt
    try:
        raw = generate_with_prompt(prompt)
        claims = _parse_claims_json(raw)
        if claims:
            logger.info(f"Extracted {len(claims)} claims")
            # Ensure claim_ids are sequential
            for i, c in enumerate(claims, 1):
                c["claim_id"] = i
                c.setdefault("type", "factual")
            return claims
    except Exception as e:
        logger.error(f"Claim extraction attempt 1 failed: {e}")

    # Retry once
    try:
        raw = generate_with_prompt(prompt)
        claims = _parse_claims_json(raw)
        if claims:
            for i, c in enumerate(claims, 1):
                c["claim_id"] = i
                c.setdefault("type", "factual")
            logger.info(f"Extracted {len(claims)} claims (retry)")
            return claims
    except Exception as e:
        logger.error(f"Claim extraction attempt 2 failed: {e}")

    # Final fallback: sentence split on original text
    logger.warning("Using sentence-split fallback for claim extraction")
    claims = _sentence_split_fallback(text)
    for i, c in enumerate(claims, 1):
        c["claim_id"] = i
    return claims
