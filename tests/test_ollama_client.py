import io
import json

import pytest

from audiobook_creator.process.llm import ollama_client
from audiobook_creator.process.llm.base import LLMError, LLMUnsupported


class _FakeHTTP:
    def __init__(self):
        self.requests = []

    def __call__(self, req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        self.requests.append((url, None if isinstance(req, str) else req.data))
        if url.endswith("/api/version"):
            body = b'{"version": "0.5.0"}'
        else:
            body = json.dumps({"message": {"content": "local answer"}}).encode()
        return io.BytesIO(body)


class _BadBodyHTTP:
    """Answers the constructor's version probe, then returns a malformed chat body."""

    def __init__(self, body):
        self.body = body

    def __call__(self, req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        if url.endswith("/api/version"):
            return io.BytesIO(b'{"version": "0.5.0"}')
        return io.BytesIO(self.body)


def test_complete_posts_chat(monkeypatch):
    fake = _FakeHTTP()
    monkeypatch.setattr(ollama_client.request, "urlopen", fake)
    client = ollama_client.OllamaClient()
    assert client.complete("hello", system="be terse") == "local answer"
    url, data = fake.requests[-1]
    payload = json.loads(data)
    assert url.endswith("/api/chat")
    assert payload["stream"] is False
    assert payload["messages"][0] == {"role": "system", "content": "be terse"}


def test_unreachable_raises(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(ollama_client.request, "urlopen", boom)
    with pytest.raises(LLMError, match="Ollama"):
        ollama_client.OllamaClient()


def test_vision_unsupported(monkeypatch, tmp_path):
    fake = _FakeHTTP()
    monkeypatch.setattr(ollama_client.request, "urlopen", fake)
    client = ollama_client.OllamaClient()
    with pytest.raises(LLMUnsupported):
        client.describe_image(tmp_path / "x.png", "describe")


def test_non_json_body_raises_llm_error(monkeypatch):
    monkeypatch.setattr(
        ollama_client.request, "urlopen", _BadBodyHTTP(b"<html>gateway error</html>")
    )
    client = ollama_client.OllamaClient()
    with pytest.raises(LLMError, match="Ollama call failed"):
        client.complete("x")


@pytest.mark.parametrize("body", [b"null", b'"hi"', b"[]"])
def test_non_object_json_body_raises_llm_error(monkeypatch, body):
    monkeypatch.setattr(ollama_client.request, "urlopen", _BadBodyHTTP(body))
    client = ollama_client.OllamaClient()
    with pytest.raises(LLMError, match="Ollama call failed"):
        client.complete("x")
