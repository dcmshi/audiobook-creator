import hashlib
import logging

from audiobook_creator.core.job import Job
from audiobook_creator.synthesize.base import TTSBackend, chunk_text, get_backend, write_wav

logger = logging.getLogger(__name__)

PAUSE = "[[pause]]"
_PAUSE_SECONDS = 0.7
_FAIL_SILENCE_SECONDS = 0.3
_RETRIES = 2


def _silence(seconds: float, sample_rate: int) -> bytes:
    return b"\x00\x00" * int(seconds * sample_rate)


def _synth_chunk(backend: TTSBackend, text: str, voice: str) -> bytes:
    last_exc: Exception | None = None
    for _attempt in range(_RETRIES + 1):
        try:
            return backend.synthesize(text, voice)
        except Exception as exc:  # noqa: BLE001 - backend errors are non-fatal by spec
            last_exc = exc
    logger.warning(
        "TTS failed after %d attempts, inserting silence for chunk %r: %s",
        _RETRIES + 1,
        text[:60],
        last_exc,
    )
    return _silence(_FAIL_SILENCE_SECONDS, backend.sample_rate)


def run_stage(job: Job) -> None:
    cfg = job.state.config
    backend = get_backend(cfg.tts_backend, local_only=cfg.local_only)
    used = f"tts:{backend.name}"
    if used not in job.state.backends_used:
        job.state.backends_used.append(used)
        job.save()

    cache_dir = job.audio_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for txt_path in sorted(job.processed_dir.glob("*.txt")):
        wav_path = job.audio_dir / f"{txt_path.stem}.wav"
        if wav_path.exists():
            continue
        pcm = bytearray()
        segments = txt_path.read_text(encoding="utf-8").split(PAUSE)
        for i, segment in enumerate(segments):
            for chunk in chunk_text(segment):
                key = hashlib.sha1(f"{backend.name}|{cfg.voice}|{chunk}".encode()).hexdigest()
                cached = cache_dir / f"{key}.pcm"
                if cached.exists():
                    data = cached.read_bytes()
                else:
                    data = _synth_chunk(backend, chunk, cfg.voice)
                    cached.write_bytes(data)
                pcm += data
            if i < len(segments) - 1:
                pcm += _silence(_PAUSE_SECONDS, backend.sample_rate)
        if not pcm:
            logger.warning("chapter %s produced no audio, skipping", txt_path.stem)
            continue
        write_wav(wav_path, bytes(pcm), backend.sample_rate)
