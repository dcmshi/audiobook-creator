import hashlib
from pathlib import Path

from audiobook_creator.process.llm.base import LLMClient


def _cache_key(parts: list[str]) -> str:
    """Hash length-prefixed components so no field's content can shift a boundary.

    A plain "|".join lets system="a|b"/user="c" and system="a"/user="b|c" produce the same
    digest, which would serve one prompt's answer for the other.
    """
    return hashlib.sha1("".join(f"{len(p)}:{p}" for p in parts).encode()).hexdigest()


def cached_complete(
    client: LLMClient,
    cache_dir: Path,
    user: str,
    *,
    system: str | None = None,
    max_tokens: int = 2048,
) -> str:
    model = getattr(client, "model", "")
    # max_tokens is part of the key: it changes the response, so raising the budget after a
    # truncation must not keep serving the truncated text.
    key = _cache_key([client.name, model, system or "", str(max_tokens), user])
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.txt"
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
