from pathlib import Path

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


@pytest.mark.kokoro
def test_real_synthesis_produces_audio():
    # Requires real model files under ./models (or ABC_MODELS_DIR). Run with: -m kokoro
    backend = KokoroBackend()
    pcm = backend.synthesize("Hello world.", "af_heart")
    assert len(pcm) > 24000  # > 0.5s of 16-bit audio
