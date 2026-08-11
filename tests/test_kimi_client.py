import io
import json
from pathlib import Path

import pytest

from audiobook_creator.process.llm import kimi_client
from audiobook_creator.process.llm.base import LLMError


class _FakeHTTP:
    def __init__(self, content="kimi answer"):
        self.requests = []
        self.content = content

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        body = json.dumps({"choices": [{"message": {"content": self.content}}]}).encode()
        return io.BytesIO(body)


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test")


def test_complete_posts_chat_completions(monkeypatch, api_key):
    fake = _FakeHTTP()
    monkeypatch.setattr(kimi_client.request, "urlopen", fake)
    client = kimi_client.KimiClient()
    assert client.complete("hello", system="be terse") == "kimi answer"
    req = fake.requests[-1]
    assert req.full_url.endswith("/chat/completions")
    assert req.headers["Authorization"] == "Bearer sk-test"
    payload = json.loads(req.data)
    assert payload["model"] == "kimi-k2.6"
    assert payload["messages"][0] == {"role": "system", "content": "be terse"}
    assert "temperature" not in payload


def test_no_key_raises(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    with pytest.raises(LLMError, match="MOONSHOT_API_KEY"):
        kimi_client.KimiClient()


def test_model_env_override(monkeypatch, api_key):
    fake = _FakeHTTP()
    monkeypatch.setattr(kimi_client.request, "urlopen", fake)
    monkeypatch.setenv("ABC_KIMI_MODEL", "kimi-k3")
    client = kimi_client.KimiClient()
    client.complete("x")
    assert json.loads(fake.requests[-1].data)["model"] == "kimi-k3"


def test_describe_image_sends_data_uri(monkeypatch, api_key, tmp_path: Path):
    fake = _FakeHTTP(content="a bar chart")
    monkeypatch.setattr(kimi_client.request, "urlopen", fake)
    png = tmp_path / "fig.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfakebytes")
    client = kimi_client.KimiClient()
    assert client.describe_image(png, "describe") == "a bar chart"
    blocks = json.loads(fake.requests[-1].data)["messages"][0]["content"]
    assert blocks[0]["type"] == "image_url"
    assert blocks[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert blocks[1] == {"type": "text", "text": "describe"}


def test_empty_content_raises(monkeypatch, api_key):
    fake = _FakeHTTP(content="")
    monkeypatch.setattr(kimi_client.request, "urlopen", fake)
    client = kimi_client.KimiClient()
    with pytest.raises(LLMError, match="no text"):
        client.complete("x")


def test_non_json_body_raises_llm_error(monkeypatch, api_key):
    def fake_urlopen(req, timeout=None):
        return io.BytesIO(b"<html>gateway error</html>")

    monkeypatch.setattr(kimi_client.request, "urlopen", fake_urlopen)
    client = kimi_client.KimiClient()
    with pytest.raises(LLMError, match="Kimi call failed"):
        client.complete("x")


@pytest.mark.parametrize("body", [b"null", b'"hi"', b"[]"])
def test_non_object_json_body_raises_llm_error(monkeypatch, api_key, body):
    def fake_urlopen(req, timeout=None):
        return io.BytesIO(body)

    monkeypatch.setattr(kimi_client.request, "urlopen", fake_urlopen)
    client = kimi_client.KimiClient()
    with pytest.raises(LLMError, match="Kimi call failed"):
        client.complete("x")


@pytest.mark.parametrize(
    "body",
    [
        b'{"choices": "x"}',
        b'{"choices": [42]}',
        b'{"choices": [{"message": "s"}]}',
        b'{"choices": [{"message": {"content": 42}}]}',
    ],
)
def test_malformed_choice_shape_raises_llm_error(monkeypatch, api_key, body):
    """Well-formed JSON object, wrong types inside: must degrade, not raise AttributeError."""

    def fake_urlopen(req, timeout=None):
        return io.BytesIO(body)

    monkeypatch.setattr(kimi_client.request, "urlopen", fake_urlopen)
    client = kimi_client.KimiClient()
    with pytest.raises(LLMError):
        client.complete("x")


@pytest.mark.llm_live
def test_live_complete_smoke():
    client = kimi_client.KimiClient()
    out = client.complete("Reply with exactly: pong")
    assert "pong" in out.lower()


def test_schemeless_url_raises_llm_error(monkeypatch, api_key):
    """Request() construction can raise ValueError; it belongs inside the containment."""
    monkeypatch.setenv("ABC_KIMI_URL", "api.moonshot.ai/v1")  # no scheme
    client = kimi_client.KimiClient()
    with pytest.raises(LLMError, match="Kimi call failed"):
        client.complete("x")


def test_unreadable_image_raises_llm_error(monkeypatch, api_key, tmp_path: Path):
    def fake_urlopen(req, timeout=None):
        raise AssertionError("must fail before any request")

    monkeypatch.setattr(kimi_client.request, "urlopen", fake_urlopen)
    client = kimi_client.KimiClient()
    with pytest.raises(LLMError, match="Kimi"):
        client.describe_image(tmp_path / "missing.png", "describe")
