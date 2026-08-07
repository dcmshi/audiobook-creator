import os
from pathlib import Path

import numpy as np

from audiobook_creator.synthesize.base import register_backend

MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"


def models_dir() -> Path:
    return Path(os.environ.get("ABC_MODELS_DIR", "models"))


def model_files_status() -> str:
    missing = [f for f in (MODEL_FILE, VOICES_FILE) if not (models_dir() / f).is_file()]
    if not missing:
        return "OK"
    return (
        f"missing {', '.join(missing)} in {models_dir()}/ - download from "
        "https://github.com/thewh1teagle/kokoro-onnx/releases (or set ABC_MODELS_DIR)"
    )


class KokoroBackend:
    name = "kokoro"
    sample_rate = 24000

    def __init__(self) -> None:
        onnx_path = models_dir() / MODEL_FILE
        voices_path = models_dir() / VOICES_FILE
        if not onnx_path.is_file() or not voices_path.is_file():
            raise RuntimeError(model_files_status())
        # Imported lazily: constructing the backend is what needs onnxruntime, and the
        # registry must stay importable on a machine with no model files.
        from kokoro_onnx import Kokoro

        self._kokoro = Kokoro(str(onnx_path), str(voices_path))

    def synthesize(self, text: str, voice: str) -> bytes:
        samples, sample_rate = self._kokoro.create(text, voice=voice, speed=1.0)
        if sample_rate != self.sample_rate:
            raise RuntimeError(f"unexpected kokoro sample rate {sample_rate}")
        clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
        return (clipped * 32767).astype("<i2").tobytes()


register_backend("kokoro", KokoroBackend, is_local=True)
