from pathlib import Path
from types import SimpleNamespace

import pytest

from audiobook_creator.process.llm.anthropic_client import AnthropicClient
from audiobook_creator.process.llm.base import LLMError


class _FakeMessages:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def _client_with(response):
    fake_sdk = SimpleNamespace(messages=_FakeMessages(response))
    return AnthropicClient(client=fake_sdk), fake_sdk


def _text_response(text):
    return SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text=text)],
    )


def test_complete_returns_text_and_caches_system():
    client, sdk = _client_with(_text_response("cleaned text"))
    out = client.complete("raw text", system="normalize this")
    assert out == "cleaned text"
    call = sdk.messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in call and "thinking" not in call


def test_refusal_raises_llm_error():
    refusal = SimpleNamespace(stop_reason="refusal", content=[])
    client, _ = _client_with(refusal)
    with pytest.raises(LLMError, match="refus"):
        client.complete("x")


def test_model_env_override(monkeypatch):
    monkeypatch.setenv("ABC_LLM_MODEL", "claude-haiku-4-5")
    client, sdk = _client_with(_text_response("ok"))
    client.complete("x")
    assert sdk.messages.calls[0]["model"] == "claude-haiku-4-5"


def test_no_credentials_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(LLMError, match="credential"):
        AnthropicClient()


def test_describe_image_sends_base64(tmp_path: Path):
    png = tmp_path / "fig.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfakebytes")
    client, sdk = _client_with(_text_response("a bar chart"))
    out = client.describe_image(png, "describe")
    assert out == "a bar chart"
    blocks = sdk.messages.calls[0]["messages"][0]["content"]
    assert blocks[0]["type"] == "image"
    assert blocks[0]["source"]["media_type"] == "image/png"


@pytest.mark.llm_live
def test_live_complete_smoke():
    client = AnthropicClient()
    out = client.complete("Reply with exactly: pong")
    assert "pong" in out.lower()
