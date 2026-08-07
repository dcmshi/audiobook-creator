import os
from pathlib import Path

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
