import hashlib
from pathlib import Path

from audiobook_creator.process.llm.base import LLMClient


def _cache_key(parts: list[str]) -> str:
    """Hash length-prefixed components so no field's content can shift a boundary.

    A plain "|".join lets system="a|b"/user="c" and system="a"/user="b|c" produce the same
    digest, which would serve one prompt's answer for the other.
    """
    return hashlib.sha1("".join(f"{len(p)}:{p}" for p in parts).encode()).hexdigest()


def _entry_path(
    client: LLMClient, cache_dir: Path, user: str, system: str | None, max_tokens: int
) -> Path:
    """The file one request maps to. Single source of truth: a second copy of this formula
    would let `evict` miss the entry `cached_complete` wrote."""
    model = getattr(client, "model", "")
    # max_tokens is part of the key: it changes the response, so raising the budget after a
    # truncation must not keep serving the truncated text.
    key = _cache_key([client.name, model, system or "", str(max_tokens), user])
    return cache_dir / f"{key}.txt"


def evict(
    client: LLMClient,
    cache_dir: Path,
    user: str,
    *,
    system: str | None = None,
    max_tokens: int = 2048,
) -> None:
    """Drop the entry a request maps to, for a caller that judged the response unusable.

    The cache cannot make that judgement itself without taking a validator argument, which
    would pull mode-specific rules in here. Instead the caller that knows the rules says so.
    """
    _entry_path(client, cache_dir, user, system, max_tokens).unlink(missing_ok=True)


def cached_complete(
    client: LLMClient,
    cache_dir: Path,
    user: str,
    *,
    system: str | None = None,
    max_tokens: int = 2048,
) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _entry_path(client, cache_dir, user, system, max_tokens)
    if path.exists():
        return path.read_text(encoding="utf-8")
    # Deliberately outside any try: an LLMError must propagate with nothing written, so a
    # transient failure is retried next run instead of being served from cache forever.
    text = client.complete(user, system=system, max_tokens=max_tokens)
    # Write aside and rename: a run killed mid-write must not leave a short file that every
    # later run accepts as a complete hit. Same idiom as JobDir.save.
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    return text
