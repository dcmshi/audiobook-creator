import http.client
import json
import os
from pathlib import Path
from urllib import request

from audiobook_creator.process.llm.base import LLMError, LLMUnsupported


class OllamaClient:
    name = "ollama"

    def __init__(self) -> None:
        self.base = os.environ.get("ABC_OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.model = os.environ.get("ABC_OLLAMA_MODEL", "qwen3:14b")
        try:
            with request.urlopen(f"{self.base}/api/version", timeout=2):
                pass
        except OSError as exc:
            raise LLMError(
                f"Ollama not reachable at {self.base} (is it running? set ABC_OLLAMA_URL)"
            ) from exc

    def complete(self, user: str, *, system: str | None = None, max_tokens: int = 2048) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                # Thinking models (qwen3) otherwise spend num_predict on reasoning and return
                # empty content. The pipeline needs speakable prose, not reasoning.
                "think": False,
                "options": {"num_predict": max_tokens},
            }
        ).encode("utf-8")
        req = request.Request(
            f"{self.base}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        # HTTPError subclasses OSError; a non-JSON or mis-encoded body raises ValueError; a
        # truncated one raises HTTPException. All must degrade to the rule-based path.
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise LLMError(f"Ollama call failed: {exc}") from exc
        if not isinstance(data, dict):  # valid JSON, wrong shape: `null`, a bare string, a list
            raise LLMError(
                f"Ollama call failed: response body was {type(data).__name__}, not an object"
            )
        text = (data.get("message") or {}).get("content", "")
        if not text.strip():
            raise LLMError("Ollama returned no text")
        return text

    def describe_image(self, image_path: Path, prompt: str, *, max_tokens: int = 1024) -> str:
        raise LLMUnsupported(
            "ollama client has no vision support; figure captions are used instead"
        )
