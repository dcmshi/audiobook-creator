import re
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class PrivacyError(RuntimeError):
    """Raised when a job flagged local_only requests a network backend."""


class TTSBackend(Protocol):
    name: str
    sample_rate: int

    def synthesize(self, text: str, voice: str) -> bytes:
        """Return raw mono 16-bit little-endian PCM at self.sample_rate."""
        ...


_REGISTRY: dict[str, tuple[Callable[[], TTSBackend], bool]] = {}


def register_backend(name: str, factory: Callable[[], TTSBackend], is_local: bool) -> None:
    _REGISTRY[name] = (factory, is_local)


def check_backend(name: str, local_only: bool = False) -> None:
    """Validate the name and the privacy gate without constructing the backend.

    Separate from get_backend so a preflight can reject a bad choice without paying to
    load a TTS model.
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown TTS backend {name!r}; known: {sorted(_REGISTRY)}")
    if local_only and not _REGISTRY[name][1]:
        raise PrivacyError(
            f"backend {name!r} sends text to a network service, but this job is local_only"
        )


def get_backend(name: str, local_only: bool = False) -> TTSBackend:
    check_backend(name, local_only)
    return _REGISTRY[name][0]()


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    out: list[str] = []
    while len(sentence) > max_chars:
        cut = sentence.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        out.append(sentence[:cut].strip())
        sentence = sentence[cut:].strip()
    if sentence:
        out.append(sentence)
    return out


def chunk_text(text: str, max_chars: int = 400) -> list[str]:
    text = text.strip()
    if not text:
        return []
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        if sentence:
            pieces.extend(_hard_split(sentence, max_chars))
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= max_chars:
            current = f"{current} {piece}"
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()
