import base64
import http.client
import json
import os
from pathlib import Path
from urllib import request

from audiobook_creator.process.llm.base import LLMError

_DEFAULT_MODEL = "kimi-k2.6"
_MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


class KimiClient:
    name = "kimi"

    def __init__(self) -> None:
        key = os.environ.get("MOONSHOT_API_KEY", "").strip()
        if not key:
            raise LLMError("no Kimi credentials found (set MOONSHOT_API_KEY, or drop --llm kimi)")
        self._key = key
        self.base = os.environ.get("ABC_KIMI_URL", "https://api.moonshot.ai/v1").rstrip("/")
        self.model = os.environ.get("ABC_KIMI_MODEL", _DEFAULT_MODEL)

    def _chat(self, messages: list[dict], max_tokens: int) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": False,
            }
        ).encode("utf-8")
        req = request.Request(
            f"{self.base}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        # HTTPError subclasses OSError (auth, rate limits); a non-JSON or mis-encoded body raises
        # ValueError; a truncated one raises HTTPException. All must degrade to the rule-based path.
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise LLMError(f"Kimi call failed: {exc}") from exc
        if not isinstance(data, dict):  # valid JSON, wrong shape: `null`, a bare string, a list
            raise LLMError(
                f"Kimi call failed: response body was {type(data).__name__}, not an object"
            )
        # Every level is type-checked: a well-formed JSON object can still carry wrong types
        # inside ({"choices": [42]}), and an AttributeError here would kill the whole run.
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMError("Kimi call failed: response contained no choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str):
            raise LLMError(
                f"Kimi call failed: malformed choice, content was {type(text).__name__}"
            )
        if not text.strip():
            raise LLMError("Kimi returned no text")
        return text

    def complete(self, user: str, *, system: str | None = None, max_tokens: int = 2048) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return self._chat(messages, max_tokens)

    def describe_image(self, image_path: Path, prompt: str, *, max_tokens: int = 1024) -> str:
        media_type = _MEDIA_TYPES.get(image_path.suffix.lower())
        if media_type is None:
            raise LLMError(f"unsupported image type: {image_path.suffix}")
        data = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
        content = [
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}},
            {"type": "text", "text": prompt},
        ]
        return self._chat([{"role": "user", "content": content}], max_tokens)
