import hashlib
import logging
import time

from audiobook_creator.core.job import Job
from audiobook_creator.synthesize.base import TTSBackend, chunk_text, get_backend, write_wav

logger = logging.getLogger(__name__)

PAUSE = "[[pause]]"
_PAUSE_SECONDS = 0.7
_FAIL_SILENCE_SECONDS = 0.3
_RETRIES = 2
# Slept before the second and third attempts so a retry can outlast a brief outage
# rather than firing all three within microseconds. Tests patch this to zeros.
_RETRY_BACKOFF_SECONDS = (0.5, 2.0)

ERROR_KEY = "synthesize:failed_chunks"


class ChunkSynthesisError(RuntimeError):
    """Every attempt at one chunk failed; the caller substitutes silence for this run."""


def _silence(seconds: float, sample_rate: int) -> bytes:
    return b"\x00\x00" * int(seconds * sample_rate)


def _synth_chunk(backend: TTSBackend, text: str, voice: str) -> bytes:
    """Return genuine backend PCM, or raise — never a silent stand-in, which must not be cached."""
    last_exc: Exception | None = None
    for attempt in range(_RETRIES + 1):
        if attempt:
            time.sleep(_RETRY_BACKOFF_SECONDS[attempt - 1])
        try:
            return backend.synthesize(text, voice)
        except Exception as exc:  # noqa: BLE001 - backend errors are non-fatal by spec
            last_exc = exc
    raise ChunkSynthesisError(
        f"TTS failed after {_RETRIES + 1} attempts for chunk {text[:60]!r}: {last_exc}"
    ) from last_exc


def run_stage(job: Job) -> None:
    cfg = job.state.config
    backend = get_backend(cfg.tts_backend, local_only=cfg.local_only)
    used = f"tts:{backend.name}"
    if used not in job.state.backends_used:
        job.state.backends_used.append(used)
        job.save()

    cache_dir = job.audio_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    failed = 0

    for txt_path in sorted(job.processed_dir.glob("*.txt")):
        wav_path = job.audio_dir / f"{txt_path.stem}.wav"
        if wav_path.exists():
            continue
        pcm = bytearray()
        chapter_total = 0
        chapter_failed = 0
        segments = txt_path.read_text(encoding="utf-8").split(PAUSE)
        for i, segment in enumerate(segments):
            for chunk in chunk_text(segment):
                chapter_total += 1
                key = hashlib.sha1(f"{backend.name}|{cfg.voice}|{chunk}".encode()).hexdigest()
                cached = cache_dir / f"{key}.pcm"
                if cached.exists():
                    pcm += cached.read_bytes()
                    continue
                try:
                    data = _synth_chunk(backend, chunk, cfg.voice)
                except ChunkSynthesisError as exc:
                    # Silence stands in for this run only. Caching it would turn a
                    # passing outage into permanent, silent damage on every later run.
                    chapter_failed += 1
                    logger.debug("%s", exc)
                    pcm += _silence(_FAIL_SILENCE_SECONDS, backend.sample_rate)
                    continue
                cached.write_bytes(data)
                pcm += data
            if i < len(segments) - 1:
                pcm += _silence(_PAUSE_SECONDS, backend.sample_rate)
        total += chapter_total
        failed += chapter_failed
        if not pcm:
            logger.warning("chapter %s produced no audio, skipping", txt_path.stem)
            continue
        if chapter_total and chapter_failed == chapter_total:
            # Writing an all-silent WAV would make the next run skip this chapter and
            # call it done. Leave it absent so a resume retries it.
            logger.warning("chapter %s failed every chunk, leaving it for a retry", txt_path.stem)
            continue
        write_wav(wav_path, bytes(pcm), backend.sample_rate)

    if failed and failed == total:
        raise RuntimeError(f"TTS produced no audio: all {total} chunks failed")
    if failed:
        message = f"{failed} of {total} chunks failed; silence inserted"
        logger.warning(message)
        job.state.errors[ERROR_KEY] = message
        job.save()
