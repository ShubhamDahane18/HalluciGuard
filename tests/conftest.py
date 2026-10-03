"""pytest configuration."""
import pytest


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Set test environment variables."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("DOCUMENTS_DIR", "data/documents")
