"""Answer generation via Ollama LLM."""
from __future__ import annotations
import requests
import logging
from typing import Dict, Any

from backend.config import settings

logger = logging.getLogger(__name__)


def _check_ollama_available() -> bool:
    """Ping Ollama to check availability."""
    try:
        resp = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _check_model_available(model: str) -> bool:
    """Check if a specific model is available in Ollama."""
    try:
        resp = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        data = resp.json()
        available_models = [m.get("name", "") for m in data.get("models", [])]
        # Normalize: strip tags for comparison
        model_base = model.split(":")[0]
        return any(model == m or model_base == m.split(":")[0] for m in available_models)
    except Exception:
        return False


def generate_answer(question: str, model: str | None = None) -> Dict[str, Any]:
    """
    Generate an answer for the given question using Ollama.

    Returns:
        {
            "question": str,
            "answer": str,
            "model": str,
        }

    Raises:
        RuntimeError: if Ollama is unavailable or model not found.
    """
    model = model or settings.ollama_model
    logger.info(f"Generating answer with model={model} for question={question!r}")

    if not _check_ollama_available():
        raise RuntimeError(
            f"Ollama is not available at {settings.ollama_base_url}. "
            "Please start Ollama with: `ollama serve`"
        )

    if not _check_model_available(model):
        raise RuntimeError(
            f"Model '{model}' is not available in Ollama. "
            f"Pull it with: `ollama pull {model}`"
        )

    prompt = (
        f"You are a knowledgeable assistant. Answer the following question accurately "
        f"and factually. Provide a detailed but concise response.\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }

    try:
        resp = requests.post(
            f"{settings.ollama_base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("response", "").strip()
        logger.info(f"Answer generated ({len(answer)} chars)")
        return {
            "question": question,
            "answer": answer,
            "model": model,
        }
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Ollama request timed out. The model may be loading - try again."
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama returned an error: {e}")


def generate_with_prompt(prompt: str, model: str | None = None) -> str:
    """
    Low-level helper: send a custom prompt to Ollama and return the text response.
    """
    model = model or settings.ollama_model

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }

    resp = requests.post(
        f"{settings.ollama_base_url}/api/generate",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()
