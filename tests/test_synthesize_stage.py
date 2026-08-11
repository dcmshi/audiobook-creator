import logging
import os
import time
from pathlib import Path

import pytest

from audiobook_creator.core.job import Job
from audiobook_creator.models import JobConfig
from audiobook_creator.synthesize import stage as synth_stage
from audiobook_creator.synthesize.base import register_backend, wav_duration_seconds


class FlakyBackend:
    """Fails permanently for chunks containing 'BAD'."""

    name = "flaky"
    sample_rate = 24000

    def synthesize(self, text: str, voice: str) -> bytes:
        if "BAD" in text:
            raise RuntimeError("synthesis exploded")
        return b"\x01\x00" * int(0.010 * len(text) * self.sample_rate)


class TransientBackend:
    """Fails its next `failures_remaining` calls, then behaves. Models a passing outage."""

    name = "transient"
    sample_rate = 24000
    failures_remaining = 0

    def synthesize(self, text: str, voice: str) -> bytes:
        if TransientBackend.failures_remaining > 0:
            TransientBackend.failures_remaining -= 1
            raise RuntimeError("temporary outage")
        return b"\x01\x00" * int(0.010 * len(text) * self.sample_rate)


register_backend("flaky", FlakyBackend, is_local=True)
register_backend("transient", TransientBackend, is_local=True)


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch):
    monkeypatch.setattr(synth_stage, "_RETRY_BACKOFF_SECONDS", (0, 0))


def _job_with_processed(tmp_path: Path, texts: dict[str, str], backend: str = "stub") -> Job:
    job = Job.create(tmp_path, JobConfig(source="x.epub", tts_backend=backend))
    for name, text in texts.items():
        (job.processed_dir / name).write_text(text, encoding="utf-8")
    return job


def test_produces_wav_per_chapter(tmp_path: Path):
    job = _job_with_processed(
        tmp_path, {"000.txt": "Hello world. [[pause]] Next section.", "001.txt": "Short."}
    )
    synth_stage.run_stage(job)
    wavs = sorted(job.audio_dir.glob("*.wav"))
    assert [w.name for w in wavs] == ["000.wav", "001.wav"]
    # pause adds 0.7s silence: chapter 0 must be longer than its text alone implies
    assert wav_duration_seconds(wavs[0]) > 0.7
    assert "tts:stub" in job.state.backends_used


def test_chunk_cache_is_populated_and_reused(tmp_path: Path):
    job = _job_with_processed(tmp_path, {"000.txt": "Hello world."})
    synth_stage.run_stage(job)
    cache_files = list((job.audio_dir / "cache").glob("*.pcm"))
    assert len(cache_files) >= 1
    mtime = cache_files[0].stat().st_mtime_ns

    (job.audio_dir / "000.wav").unlink()  # force resynthesis
    synth_stage.run_stage(job)
    assert cache_files[0].stat().st_mtime_ns == mtime  # reused, not rewritten


def test_existing_wavs_skipped(tmp_path: Path):
    job = _job_with_processed(tmp_path, {"000.txt": "Hello."})
    synth_stage.run_stage(job)
    wav = job.audio_dir / "000.wav"
    mtime = wav.stat().st_mtime_ns
    synth_stage.run_stage(job)
    assert wav.stat().st_mtime_ns == mtime


def test_edited_processed_text_rebuilds_the_chapter(tmp_path: Path):
    job = _job_with_processed(tmp_path, {"000.txt": "Hello world."})
    synth_stage.run_stage(job)
    wav = job.audio_dir / "000.wav"
    before = wav.read_bytes()

    txt = job.processed_dir / "000.txt"
    txt.write_text("Hello world. And a good deal more text than before.", encoding="utf-8")
    # Windows' coarse clock can stamp the rewrite inside the same tick as the WAV write,
    # which would make "newer" untestable; push it clearly past.
    future = time.time() + 10
    os.utime(txt, (future, future))

    synth_stage.run_stage(job)
    assert wav.read_bytes() != before


def test_failed_chunk_becomes_silence_not_crash(tmp_path: Path, caplog):
    # Pause-separated so the failing chunk is one of three: a run in which *every*
    # chunk fails is a hard error, covered separately below.
    job = _job_with_processed(
        tmp_path,
        {"000.txt": "Good text. [[pause]] BAD text. [[pause]] More good."},
        backend="flaky",
    )
    synth_stage.run_stage(job)  # must not raise
    assert (job.audio_dir / "000.wav").exists()
    assert any("BAD" in r.message or "failed" in r.message.lower() for r in caplog.records)
    assert "1 of 3" in job.state.errors["synthesize:failed_chunks"]


def test_degraded_chapter_auto_repairs_on_the_next_run(tmp_path: Path):
    job = _job_with_processed(
        tmp_path,
        {"000.txt": "This first segment is long enough to matter. [[pause]] Second segment here."},
        backend="transient",
    )
    TransientBackend.failures_remaining = 3  # exhausts every attempt for chunk one

    synth_stage.run_stage(job)
    cache_dir = job.audio_dir / "cache"
    assert len(list(cache_dir.glob("*.pcm"))) == 1  # only the chunk that really synthesized
    assert "1 of 2" in job.state.errors["synthesize:failed_chunks"]
    assert job.state.errors["synthesize:degraded"] == "000"
    degraded_duration = wav_duration_seconds(job.audio_dir / "000.wav")

    # Backend is healthy now. No manual deletion: the recorded degradation is what
    # makes the stage rebuild this chapter.
    synth_stage.run_stage(job)
    assert len(list(cache_dir.glob("*.pcm"))) == 2
    assert wav_duration_seconds(job.audio_dir / "000.wav") > degraded_duration
    assert not [k for k in job.state.errors if k.startswith("synthesize:")]


def test_wholly_failed_chapter_raises_and_spares_healthy_chapters(tmp_path: Path):
    job = _job_with_processed(
        tmp_path,
        {"000.txt": "Good text.", "001.txt": "BAD one. [[pause]] BAD two."},
        backend="flaky",
    )
    with pytest.raises(RuntimeError, match="chapter\\(s\\) 001"):
        synth_stage.run_stage(job)
    # The healthy chapter survives, so a resume only retries the failed one.
    assert (job.audio_dir / "000.wav").exists()
    assert not (job.audio_dir / "001.wav").exists()


def test_all_chunks_failing_raises_and_leaves_nothing_behind(tmp_path: Path):
    job = _job_with_processed(
        tmp_path, {"000.txt": "BAD one. [[pause]] BAD two."}, backend="flaky"
    )
    with pytest.raises(RuntimeError, match="all 2 chunks attempted in this run failed"):
        synth_stage.run_stage(job)
    # Nothing cached and no WAV, so a resume retries from scratch instead of
    # accepting a silent chapter as done.
    assert list((job.audio_dir / "cache").glob("*.pcm")) == []
    assert not (job.audio_dir / "000.wav").exists()


def test_clean_run_leaves_no_stale_failure_records(tmp_path: Path):
    job = _job_with_processed(tmp_path, {"000.txt": "Hello world."})
    job.state.errors["synthesize:failed_chunks"] = "3 of 4 chunks failed; silence inserted"
    job.state.errors["synthesize:degraded"] = "000"
    job.save()

    synth_stage.run_stage(job)
    assert not [k for k in job.state.errors if k.startswith("synthesize:")]
    assert (job.audio_dir / "000.wav").exists()


class FixedBackend:
    """Fixed-length PCM per call, so a duration delta isolates inserted silence."""

    name = "fixed"
    sample_rate = 24000

    def synthesize(self, text: str, voice: str) -> bytes:
        return b"\x01\x00" * 2400  # 0.1s


register_backend("fixed", FixedBackend, is_local=True)


def _voice_recorder(name: str) -> list[tuple[str, str]]:
    """Register a backend under `name` that records every (voice, text) it is asked for."""
    recorded: list[tuple[str, str]] = []

    class VoiceRecorder:
        sample_rate = 24000

        def __init__(self):
            self.name = name

        def synthesize(self, text: str, voice: str) -> bytes:
            recorded.append((voice, text))
            return b"\x01\x00" * 240

    VoiceRecorder.name = name
    register_backend(name, VoiceRecorder, is_local=True)
    return recorded


def test_speaker_tags_switch_voices(tmp_path: Path):
    recorded = _voice_recorder("rec-switch")
    job = _job_with_processed(
        tmp_path,
        {"000.txt": "[[speaker:1]] Hello there listener.\n[[speaker:2]] And hello back to you."},
        backend="rec-switch",
    )
    synth_stage.run_stage(job)
    assert [voice for voice, _ in recorded] == ["af_heart", "am_adam"]


def test_untagged_text_uses_config_voice(tmp_path: Path):
    recorded = _voice_recorder("rec-plain")
    job = _job_with_processed(
        tmp_path, {"000.txt": "Plain narration text."}, backend="rec-plain"
    )
    synth_stage.run_stage(job)
    assert (job.audio_dir / "000.wav").exists()
    assert [voice for voice, _ in recorded] == ["af_heart"]


def test_out_of_range_speaker_uses_default_voice_and_warns_once(tmp_path: Path, caplog):
    recorded = _voice_recorder("rec-range")
    job = _job_with_processed(
        tmp_path,
        {"000.txt": "[[speaker:9]] One.\n[[speaker:9]] Two.\n[[speaker:9]] Three."},
        backend="rec-range",
    )
    with caplog.at_level(logging.WARNING):
        synth_stage.run_stage(job)
    assert [voice for voice, _ in recorded] == ["af_heart"] * 3
    warnings = [r for r in caplog.records if "speaker" in r.getMessage()]
    assert len(warnings) == 1  # once per chapter, not once per line


def test_silence_separates_turns_with_different_voices(tmp_path: Path):
    two_voices = _job_with_processed(
        tmp_path / "diff",
        {"000.txt": "[[speaker:1]] One.\n[[speaker:2]] Two."},
        backend="fixed",
    )
    synth_stage.run_stage(two_voices)
    one_voice = _job_with_processed(
        tmp_path / "same",
        {"000.txt": "[[speaker:1]] One.\n[[speaker:1]] Two."},
        backend="fixed",
    )
    synth_stage.run_stage(one_voice)
    delta = wav_duration_seconds(two_voices.audio_dir / "000.wav") - wav_duration_seconds(
        one_voice.audio_dir / "000.wav"
    )
    assert delta == pytest.approx(0.3, abs=0.02)


def test_orphaned_chapter_audio_is_removed(tmp_path: Path):
    """A verbatim job re-run as a podcast must not keep the old chapters' audio."""
    job = _job_with_processed(
        tmp_path, {"000.txt": "One.", "001.txt": "Two.", "002.txt": "Three."}
    )
    synth_stage.run_stage(job)
    assert len(list(job.audio_dir.glob("*.wav"))) == 3
    for name in ("001.txt", "002.txt"):
        (job.processed_dir / name).unlink()
    (job.processed_dir / "000.txt").write_text("[[speaker:1]] Digest.", encoding="utf-8")
    synth_stage.run_stage(job)
    assert [w.name for w in sorted(job.audio_dir.glob("*.wav"))] == ["000.wav"]
    assert (job.audio_dir / "cache").is_dir()  # the chunk cache is not swept


def test_wav_sweep_skips_an_empty_processed_dir(tmp_path: Path):
    """No processed text at all means nothing to compare against, not everything orphaned."""
    job = _job_with_processed(tmp_path, {"000.txt": "One."})
    synth_stage.run_stage(job)
    (job.processed_dir / "000.txt").unlink()
    synth_stage.run_stage(job)
    assert (job.audio_dir / "000.wav").exists()
