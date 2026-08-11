import http.client
import io
import json
import logging

import pytest

from audiobook_creator.core.privacy import PrivacyError
from audiobook_creator.process import llm
from audiobook_creator.process.llm import ollama_client
from audiobook_creator.process.llm.base import LLMError, LLMUnsupported


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ollama config comes from the environment; pin it so tests read the real defaults."""
    for var in ("ABC_LLM", "ABC_OLLAMA_URL", "ABC_OLLAMA_MODEL"):
        monkeypatch.delenv(var, raising=False)


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


class _RecordingHTTP:
    """Answers the version probe and records every URL, so a test can prove none was requested."""

    def __init__(self):
        self.calls = []

    def __call__(self, req, timeout=None):
        self.calls.append(req if isinstance(req, str) else req.full_url)
        return io.BytesIO(b'{"version": "0.5.0"}')


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
    assert client.complete("hello", system="be terse", max_tokens=123) == "local answer"
    url, data = fake.requests[-1]
    payload = json.loads(data)
    assert url.endswith("/api/chat")
    assert payload["stream"] is False
    # Thinking models otherwise spend num_predict on reasoning and return empty content.
    assert payload["think"] is False
    assert payload["options"]["num_predict"] == 123
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


def test_null_content_raises_llm_error(monkeypatch):
    """`content: null` yields None from .get, which used to blow up at .strip()."""
    monkeypatch.setattr(
        ollama_client.request, "urlopen", _BadBodyHTTP(b'{"message": {"content": null}}')
    )
    client = ollama_client.OllamaClient()
    with pytest.raises(LLMError, match="no text"):
        client.complete("x")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:11434", True),
        ("http://127.0.0.1:11434", True),
        ("http://127.0.0.2:11434", True),  # all of 127.0.0.0/8 is this machine
        ("http://[::1]:11434", True),
        ("http://192.168.1.50:11434", False),  # same LAN is still off-box
        ("https://ollama.example.com", False),
        ("http://10.0.0.4:11434", False),
        ("", False),
        ("http://[::1", False),  # unbalanced bracket: urlsplit raises, must not escape
    ],
)
def test_is_local_endpoint(monkeypatch, url, expected):
    monkeypatch.setenv("ABC_OLLAMA_URL", url)
    assert ollama_client.is_local_endpoint() is expected


def test_malformed_url_degrades_instead_of_crashing(monkeypatch):
    """The predicate runs before the local_only check, so a raise here crashes every job."""
    monkeypatch.setenv("ABC_OLLAMA_URL", "http://[::1")
    assert llm.resolve_llm(local_only=False, use_llm=True) is None
    with pytest.raises(PrivacyError):  # unparseable is treated as remote, not as a crash
        llm.resolve_llm(local_only=True, use_llm=True)


def test_local_only_refuses_remote_ollama_before_any_request(monkeypatch):
    recorder = _RecordingHTTP()
    monkeypatch.setattr(ollama_client.request, "urlopen", recorder)
    monkeypatch.setenv("ABC_OLLAMA_URL", "http://192.168.1.50:11434")
    with pytest.raises(PrivacyError):
        llm.resolve_llm(local_only=True, use_llm=True)
    assert recorder.calls == []  # the version probe would itself have shipped a packet


def test_local_only_allows_loopback_ollama(monkeypatch):
    recorder = _RecordingHTTP()
    monkeypatch.setattr(ollama_client.request, "urlopen", recorder)
    client = llm.resolve_llm(local_only=True, use_llm=True)
    assert client is not None
    assert client.name == "ollama"
    assert recorder.calls == ["http://localhost:11434/api/version"]


def test_empty_url_raises_llm_error(monkeypatch):
    # Real urlopen: base "" makes the probe URL "/api/version" -> ValueError, no network touched.
    monkeypatch.setenv("ABC_OLLAMA_URL", "")
    with pytest.raises(LLMError, match="Ollama not reachable"):
        ollama_client.OllamaClient()


def test_non_http_service_on_port_raises_llm_error(monkeypatch):
    def bad_status(req, timeout=None):
        raise http.client.BadStatusLine("not-http-garbage")

    monkeypatch.setattr(ollama_client.request, "urlopen", bad_status)
    with pytest.raises(LLMError, match="Ollama not reachable"):
        ollama_client.OllamaClient()


def test_payload_declares_the_context_window(monkeypatch):
    fake = _FakeHTTP()
    monkeypatch.setattr(ollama_client.request, "urlopen", fake)
    monkeypatch.setenv("ABC_OLLAMA_NUM_CTX", "4096")
    client = ollama_client.OllamaClient()
    client.complete("x")
    _url, data = fake.requests[-1]
    assert json.loads(data)["options"]["num_ctx"] == 4096


def test_default_context_window_is_declared(monkeypatch):
    fake = _FakeHTTP()
    monkeypatch.setattr(ollama_client.request, "urlopen", fake)
    client = ollama_client.OllamaClient()
    client.complete("x")
    _url, data = fake.requests[-1]
    assert json.loads(data)["options"]["num_ctx"] == 8192


def test_prompt_larger_than_the_context_window_warns(monkeypatch, caplog):
    fake = _FakeHTTP()
    monkeypatch.setattr(ollama_client.request, "urlopen", fake)
    monkeypatch.setenv("ABC_OLLAMA_NUM_CTX", "1024")
    client = ollama_client.OllamaClient()
    with caplog.at_level(logging.WARNING):
        client.complete("word " * 2000)  # ~10000 chars -> ~3333 tokens
    assert "1024" in caplog.text  # names the window
    assert any("token" in r.getMessage() for r in caplog.records)  # and the estimate


def test_prompt_within_the_context_window_is_quiet(monkeypatch, caplog):
    fake = _FakeHTTP()
    monkeypatch.setattr(ollama_client.request, "urlopen", fake)
    client = ollama_client.OllamaClient()
    with caplog.at_level(logging.WARNING):
        client.complete("a short prompt")
    assert caplog.text == ""


def test_generation_budget_counts_against_the_context_window(monkeypatch, caplog):
    """num_ctx bounds prompt plus generation: podcast's 16k budget alone overflows the default."""
    fake = _FakeHTTP()
    monkeypatch.setattr(ollama_client.request, "urlopen", fake)
    client = ollama_client.OllamaClient()
    with caplog.at_level(logging.WARNING):
        client.complete("a short source", max_tokens=16000)
    assert "8192" in caplog.text
    assert any("token" in r.getMessage() for r in caplog.records)
