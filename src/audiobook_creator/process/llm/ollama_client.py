import http.client
import ipaddress
import json
import logging
import os
from pathlib import Path
from urllib import request
from urllib.parse import urlparse

from audiobook_creator.process.llm.base import LLMError, LLMUnsupported

_DEFAULT_URL = "http://localhost:11434"
# Ollama's own default is small and applied invisibly; 8192 matches what the shipped models
# handle comfortably. A conservative 3 chars per token keeps the estimate on the safe side.
_DEFAULT_NUM_CTX = 8192
_CHARS_PER_TOKEN = 3

logger = logging.getLogger(__name__)


def base_url() -> str:
    return os.environ.get("ABC_OLLAMA_URL", _DEFAULT_URL).rstrip("/")


def _num_ctx() -> int:
    try:
        return int(os.environ.get("ABC_OLLAMA_NUM_CTX", _DEFAULT_NUM_CTX))
    except ValueError:
        logger.warning("ignoring non-numeric ABC_OLLAMA_NUM_CTX; using %d", _DEFAULT_NUM_CTX)
        return _DEFAULT_NUM_CTX


def is_local_endpoint() -> bool:
    """True when the configured Ollama URL is this machine.

    Registered as ollama's locality predicate so a local_only job refuses a LAN or hosted
    Ollama before any request is made. "Local" means this machine, not this network: a
    host on the same LAN is still off-box. Any hostname other than `localhost` is treated
    as remote — resolving it is itself a network action, and DNS can point anywhere.
    """
    try:
        host = urlparse(base_url()).hostname
        if host is None:
            return False
        if host == "localhost":
            return True
        return ipaddress.ip_address(host).is_loopback
    except ValueError:  # unparseable URL, or a host that is not an IP literal
        return False


class OllamaClient:
    name = "ollama"

    def __init__(self) -> None:
        self.base = base_url()
        self.model = os.environ.get("ABC_OLLAMA_MODEL", "qwen3:14b")
        try:
            with request.urlopen(f"{self.base}/api/version", timeout=2):
                pass
        # A malformed URL raises ValueError and a non-HTTP service on the port raises
        # HTTPException; neither is an OSError, and both must degrade, not crash.
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise LLMError(
                f"Ollama not reachable at {self.base} (is it running? set ABC_OLLAMA_URL)"
            ) from exc

    def complete(self, user: str, *, system: str | None = None, max_tokens: int = 2048) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        # Declared rather than left to the server default: Ollama silently drops whatever does
        # not fit, so an undeclared window turns a too-long chapter into quietly missing text.
        num_ctx = _num_ctx()
        # num_ctx bounds the prompt AND the generation, so the output budget counts: a 16k
        # podcast script overflows the 8192 default on its own, whatever the source size.
        estimated_tokens = sum(len(m["content"]) for m in messages) // _CHARS_PER_TOKEN
        if estimated_tokens + max_tokens > num_ctx:
            logger.warning(
                "prompt (~%d tokens) plus the %d-token output budget exceeds the Ollama context "
                "window of %d; the overflow is dropped silently. Raise ABC_OLLAMA_NUM_CTX, or "
                "lower the budget.",
                estimated_tokens,
                max_tokens,
                num_ctx,
            )
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                # Thinking models (qwen3) otherwise spend num_predict on reasoning and return
                # empty content. The pipeline needs speakable prose, not reasoning.
                "think": False,
                "options": {"num_predict": max_tokens, "num_ctx": num_ctx},
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
        # `or ""`, not a .get default: a `content: null` body yields None, which the default
        # would not catch.
        text = (data.get("message") or {}).get("content") or ""
        if not text.strip():
            raise LLMError("Ollama returned no text")
        return text

    def describe_image(self, image_path: Path, prompt: str, *, max_tokens: int = 1024) -> str:
        raise LLMUnsupported(
            "ollama client has no vision support; figure captions are used instead"
        )
