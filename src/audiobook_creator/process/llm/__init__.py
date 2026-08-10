import contextlib
import logging
import os
from collections.abc import Callable

from audiobook_creator.core.privacy import PrivacyError
from audiobook_creator.process.llm.base import LLMClient, LLMError

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, tuple[Callable[[], LLMClient], bool]] = {}


def register_llm(name: str, factory: Callable[[], LLMClient], is_local: bool) -> None:
    _REGISTRY[name] = (factory, is_local)


def _candidates(provider: str | None) -> list[str]:
    forced = (provider or os.environ.get("ABC_LLM", "")).strip().lower()
    if forced == "none":
        return []
    if forced:
        return [forced]
    return ["ollama"]  # paid providers are opt-in only (--llm / ABC_LLM)


def resolve_llm(
    *, local_only: bool, use_llm: bool, provider: str | None = None
) -> LLMClient | None:
    if not use_llm:
        return None
    for name in _candidates(provider):
        if name not in _REGISTRY:
            logger.warning("unknown LLM provider %r requested; skipping", name)
            continue
        factory, is_local = _REGISTRY[name]
        if local_only and not is_local:
            raise PrivacyError(
                f"LLM provider {name!r} sends text to a network service, "
                "but this job is local_only"
            )
        try:
            return factory()
        except LLMError as exc:
            logger.info("LLM provider %r unavailable: %s", name, exc)
    return None


def _register_builtin() -> None:
    from audiobook_creator.process.llm.anthropic_client import AnthropicClient
    from audiobook_creator.process.llm.kimi_client import KimiClient
    from audiobook_creator.process.llm.ollama_client import OllamaClient

    register_llm("anthropic", AnthropicClient, is_local=False)
    register_llm("kimi", KimiClient, is_local=False)
    register_llm("ollama", OllamaClient, is_local=True)


# The client modules land in Tasks 2-4; the package must import before they exist.
with contextlib.suppress(ImportError):
    _register_builtin()
