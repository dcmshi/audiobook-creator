import hashlib
from pathlib import Path

from audiobook_creator.process.llm.base import LLMClient


def cached_complete(
    client: LLMClient,
    cache_dir: Path,
    user: str,
    *,
    system: str | None = None,
    max_tokens: int = 2048,
) -> str:
    model = getattr(client, "model", "")
    key = hashlib.sha1(
        f"{client.name}|{model}|{system or ''}|{user}".encode()
    ).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    # Deliberately outside any try: an LLMError must propagate with nothing written, so a
    # transient failure is retried next run instead of being served from cache forever.
    text = client.complete(user, system=system, max_tokens=max_tokens)
    path.write_text(text, encoding="utf-8")
    return text
