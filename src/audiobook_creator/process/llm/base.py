from pathlib import Path
from typing import Protocol


class LLMError(RuntimeError):
    """A provider could not be constructed or a call failed."""


class LLMUnsupported(LLMError):
    """The provider cannot perform the requested operation (e.g. vision)."""


class LLMClient(Protocol):
    name: str

    def complete(self, user: str, *, system: str | None = None, max_tokens: int = 2048) -> str: ...

    def describe_image(self, image_path: Path, prompt: str, *, max_tokens: int = 1024) -> str: ...
