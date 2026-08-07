from pathlib import Path

import pytest

from audiobook_creator.synthesize.base import (
    PrivacyError,
    chunk_text,
    get_backend,
    register_backend,
    wav_duration_seconds,
    write_wav,
)
from audiobook_creator.synthesize.stub import StubBackend


def test_chunk_respects_sentences():
    text = "First sentence here. Second one is also short. Third."
    chunks = chunk_text(text, max_chars=45)
    assert all(len(c) <= 45 for c in chunks)
    assert all(c.endswith((".", "!", "?")) for c in chunks)  # sentence boundaries only
    assert " ".join(chunks) == text


def test_chunk_hard_splits_monster_sentence():
    text = "word " * 200  # one 1000-char "sentence"
    chunks = chunk_text(text.strip(), max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) >= 9


def test_chunk_empty_returns_empty():
    assert chunk_text("   ") == []


def test_stub_backend_duration_scales_with_text():
    stub = StubBackend()
    short = stub.synthesize("hi", "any")
    long = stub.synthesize("hello there friend", "any")
    assert len(long) > len(short)
    assert len(short) % 2 == 0  # 16-bit samples


def test_registry_returns_stub():
    backend = get_backend("stub")
    assert backend.name == "stub"


def test_registry_unknown_name():
    with pytest.raises(ValueError, match="unknown TTS backend"):
        get_backend("nope")


def test_privacy_blocks_network_backends():
    register_backend("fake-cloud", StubBackend, is_local=False)
    with pytest.raises(PrivacyError):
        get_backend("fake-cloud", local_only=True)
    assert get_backend("fake-cloud", local_only=False).name == "stub"


def test_wav_roundtrip(tmp_path: Path):
    pcm = b"\x00\x01" * 24000  # exactly 1 second at 24 kHz mono 16-bit
    path = tmp_path / "t.wav"
    write_wav(path, pcm, 24000)
    assert abs(wav_duration_seconds(path) - 1.0) < 0.001
