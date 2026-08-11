import base64
import os
from pathlib import Path

from audiobook_creator.process.llm.base import LLMError

_DEFAULT_MODEL = "claude-opus-5"
_MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


class AnthropicClient:
    name = "anthropic"

    def __init__(self, client=None) -> None:
        self.model = os.environ.get("ABC_LLM_MODEL", _DEFAULT_MODEL)
        if client is not None:
            self._client = client
            return
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise LLMError(
                "no Anthropic credentials found (set ANTHROPIC_API_KEY, or drop --llm anthropic)"
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dep is in base install
            raise LLMError("anthropic SDK not installed") from exc
        self._client = anthropic.Anthropic()

    def _create(self, *, system: str | None, content, max_tokens: int) -> str:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            kwargs["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise LLMError(f"anthropic call failed: {exc}") from exc
        if response.stop_reason == "refusal":
            raise LLMError("anthropic refused the request (stop_reason=refusal)")
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        if not text.strip():
            raise LLMError("anthropic returned no text")
        return text

    def complete(self, user: str, *, system: str | None = None, max_tokens: int = 2048) -> str:
        return self._create(system=system, content=user, max_tokens=max_tokens)

    def describe_image(self, image_path: Path, prompt: str, *, max_tokens: int = 1024) -> str:
        media_type = _MEDIA_TYPES.get(image_path.suffix.lower())
        if media_type is None:
            raise LLMError(f"unsupported image type: {image_path.suffix}")
        # A figure recorded at ingest can be gone or unreadable by the time it is described;
        # that is this layer's failure to report, not an OSError for the caller to trip over.
        try:
            raw = image_path.read_bytes()
        except OSError as exc:
            raise LLMError(f"anthropic call failed: cannot read {image_path}: {exc}") from exc
        data = base64.standard_b64encode(raw).decode("ascii")
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
            {"type": "text", "text": prompt},
        ]
        return self._create(system=None, content=content, max_tokens=max_tokens)
