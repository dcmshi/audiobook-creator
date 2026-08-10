import pytest

from audiobook_creator.core.privacy import PrivacyError
from audiobook_creator.process import llm
from audiobook_creator.process.llm.base import LLMError


class FakeLLM:
    name = "fake"

    def complete(self, user, *, system=None, max_tokens=2048):
        return f"ok:{user}"

    def describe_image(self, image_path, prompt, *, max_tokens=1024):
        return "an image"


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    monkeypatch.setattr(llm, "_REGISTRY", dict(llm._REGISTRY))
    monkeypatch.delenv("ABC_LLM", raising=False)


def test_privacy_error_is_shared_with_tts():
    from audiobook_creator.synthesize.base import PrivacyError as TtsPrivacyError

    assert TtsPrivacyError is PrivacyError


def test_env_override_selects_provider(monkeypatch):
    llm.register_llm("fake", FakeLLM, is_local=True)
    monkeypatch.setenv("ABC_LLM", "fake")
    client = llm.resolve_llm(local_only=False, use_llm=True)
    assert client.name == "fake"


def test_use_llm_false_resolves_none():
    assert llm.resolve_llm(local_only=False, use_llm=False) is None


def test_local_only_blocks_network_provider(monkeypatch):
    llm.register_llm("fake-cloud", FakeLLM, is_local=False)
    monkeypatch.setenv("ABC_LLM", "fake-cloud")
    with pytest.raises(PrivacyError):
        llm.resolve_llm(local_only=True, use_llm=True)


def test_failing_factory_resolves_to_none(monkeypatch):
    def broken():
        raise LLMError("not running")

    monkeypatch.setattr(llm, "_REGISTRY", {"ollama": (broken, True)})
    assert llm.resolve_llm(local_only=False, use_llm=True) is None


def test_default_is_ollama_only():
    assert llm._candidates(None) == ["ollama"]


def test_job_provider_beats_env(monkeypatch):
    llm.register_llm("fake", FakeLLM, is_local=True)
    monkeypatch.setenv("ABC_LLM", "anthropic")
    client = llm.resolve_llm(local_only=False, use_llm=True, provider="fake")
    assert client.name == "fake"
