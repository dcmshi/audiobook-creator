from pathlib import Path

import numpy as np
import pytest

from audiobook_creator.synthesize.kokoro import KokoroBackend


def test_missing_models_give_actionable_error(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ABC_MODELS_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="kokoro-onnx/releases"):
        KokoroBackend()


def test_registered_as_local():
    from audiobook_creator.synthesize.base import _REGISTRY

    assert "kokoro" in _REGISTRY
    assert _REGISTRY["kokoro"][1] is True  # is_local


class _FakeKokoro:
    """Stands in for kokoro_onnx.Kokoro so the conversion is testable without model files."""

    def __init__(self, sample_rate: int = 24000):
        self._sample_rate = sample_rate

    def create(self, text, voice, speed):
        return np.array([0.0, 1.5, -1.5], dtype=np.float32), self._sample_rate


def _backend_with(fake: _FakeKokoro) -> KokoroBackend:
    # Bypass __init__ so no model files or onnxruntime are needed.
    backend = KokoroBackend.__new__(KokoroBackend)
    backend._kokoro = fake
    return backend


def test_float_samples_convert_to_clipped_little_endian_pcm():
    pcm = _backend_with(_FakeKokoro()).synthesize("hi", "af_heart")
    assert np.frombuffer(pcm, dtype="<i2").tolist() == [0, 32767, -32767]


def test_unexpected_sample_rate_is_rejected():
    backend = _backend_with(_FakeKokoro(sample_rate=22050))
    with pytest.raises(RuntimeError, match="22050"):
        backend.synthesize("hi", "af_heart")


@pytest.mark.kokoro
def test_real_synthesis_produces_audio():
    # Requires real model files under ./models (or ABC_MODELS_DIR). Run with: -m kokoro
    backend = KokoroBackend()
    pcm = backend.synthesize("Hello world.", "af_heart")
    assert len(pcm) > 24000  # > 0.5s of 16-bit audio
