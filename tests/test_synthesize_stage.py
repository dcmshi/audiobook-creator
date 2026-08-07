from pathlib import Path

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


register_backend("flaky", FlakyBackend, is_local=True)


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


def test_failed_chunk_becomes_silence_not_crash(tmp_path: Path, caplog):
    job = _job_with_processed(
        tmp_path, {"000.txt": "Good text. BAD text. More good."}, backend="flaky"
    )
    synth_stage.run_stage(job)  # must not raise
    assert (job.audio_dir / "000.wav").exists()
    assert any("BAD" in r.message or "failed" in r.message.lower() for r in caplog.records)
