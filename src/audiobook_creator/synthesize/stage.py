import hashlib
import logging
import time
from pathlib import Path

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

# This stage owns these keys: the engine only clears the bare "synthesize" key, so a
# stale record here would outlive the run that caused it unless we clear it ourselves.
FAILED_CHUNKS_KEY = "synthesize:failed_chunks"
DEGRADED_KEY = "synthesize:degraded"


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


def _recorded_stems(value: str | None) -> list[str]:
    return [stem for stem in (value or "").split(",") if stem]


def _synthesize_chapter(
    backend: TTSBackend, voice: str, cache_dir: Path, text: str
) -> tuple[bytes, int, int]:
    """Return (pcm, chunks attempted, chunks failed) for one chapter's text."""
    pcm = bytearray()
    attempted = 0
    failed = 0
    segments = text.split(PAUSE)
    for i, segment in enumerate(segments):
        for chunk in chunk_text(segment):
            attempted += 1
            key = hashlib.sha1(f"{backend.name}|{voice}|{chunk}".encode()).hexdigest()
            cached = cache_dir / f"{key}.pcm"
            if cached.exists():
                pcm += cached.read_bytes()
                continue
            try:
                data = _synth_chunk(backend, chunk, voice)
            except ChunkSynthesisError as exc:
                # Silence stands in for this run only. Caching it would turn a passing
                # outage into permanent, silent damage on every later run.
                failed += 1
                logger.debug("%s", exc)
                pcm += _silence(_FAIL_SILENCE_SECONDS, backend.sample_rate)
                continue
            cached.write_bytes(data)
            pcm += data
        if i < len(segments) - 1:
            pcm += _silence(_PAUSE_SECONDS, backend.sample_rate)
    return bytes(pcm), attempted, failed


def _record_outcome(job: Job, *, attempted: int, failed: int, degraded: list[str]) -> None:
    """Refresh this stage's records. A clean run must leave none of them behind."""
    job.state.errors.pop(FAILED_CHUNKS_KEY, None)
    job.state.errors.pop(DEGRADED_KEY, None)
    # Only claim silence was inserted when a written WAV actually contains it: a chapter
    # that lost every chunk is never written, and is reported by raising instead.
    if degraded:
        message = f"{failed} of {attempted} chunks failed; silence inserted"
        logger.warning(message)
        job.state.errors[FAILED_CHUNKS_KEY] = message
        job.state.errors[DEGRADED_KEY] = ",".join(degraded)
    job.save()


def run_stage(job: Job) -> None:
    cfg = job.state.config
    backend = get_backend(cfg.tts_backend, local_only=cfg.local_only)
    used = f"tts:{backend.name}"
    if used not in job.state.backends_used:
        job.state.backends_used.append(used)
        job.save()

    cache_dir = job.audio_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Chapters recorded degraded have failure-silence baked into their WAV, which every
    # later run would skip over. Drop those WAVs so this run rebuilds them: the cached
    # good chunks make the rebuild cheap and the failed chunks get a fresh attempt.
    for stem in _recorded_stems(job.state.errors.get(DEGRADED_KEY)):
        (job.audio_dir / f"{stem}.wav").unlink(missing_ok=True)

    attempted = 0
    failed = 0
    degraded: list[str] = []
    wholly_failed: list[str] = []

    for txt_path in sorted(job.processed_dir.glob("*.txt")):
        wav_path = job.audio_dir / f"{txt_path.stem}.wav"
        # Rebuild when the text is newer than its audio, so hand-edited processed text
        # is honoured by `abc resume --from-stage synthesize`. The chunk cache means
        # unchanged passages cost a file read rather than a fresh synthesis.
        if wav_path.exists() and wav_path.stat().st_mtime_ns >= txt_path.stat().st_mtime_ns:
            continue
        pcm, chapter_attempted, chapter_failed = _synthesize_chapter(
            backend, cfg.voice, cache_dir, txt_path.read_text(encoding="utf-8")
        )
        attempted += chapter_attempted
        failed += chapter_failed
        if not pcm:
            logger.warning("chapter %s produced no audio, skipping", txt_path.stem)
            continue
        if chapter_attempted and chapter_failed == chapter_attempted:
            # An all-silent WAV would be skipped by every later run and counted as done.
            wholly_failed.append(txt_path.stem)
            continue
        if chapter_failed:
            degraded.append(txt_path.stem)
        write_wav(wav_path, pcm, backend.sample_rate)

    _record_outcome(job, attempted=attempted, failed=failed, degraded=degraded)

    if failed and failed == attempted:
        raise RuntimeError(
            f"TTS produced no audio: all {attempted} chunks attempted in this run failed"
        )
    if wholly_failed:
        raise RuntimeError(
            f"TTS failed for chapter(s) {', '.join(wholly_failed)}: every chunk failed this run"
        )
