from pathlib import Path

from helpers import requires_ffmpeg
from typer.testing import CliRunner

from audiobook_creator.cli import app
from audiobook_creator.core.job import Job
from audiobook_creator.models import JobConfig
from audiobook_creator.package.ffmpeg import probe_chapters
from audiobook_creator.synthesize import stage as synth_stage
from audiobook_creator.synthesize.base import register_backend

runner = CliRunner()


class _ExplodingBackend:
    """Registered and local, so it clears preflight and then fails during synthesis."""

    name = "exploding"
    sample_rate = 24000

    def synthesize(self, text: str, voice: str) -> bytes:
        raise RuntimeError("backend detonated")


register_backend("exploding", _ExplodingBackend, is_local=True)


@requires_ffmpeg
def test_end_to_end_epub_to_m4b(make_epub, tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    result = runner.invoke(
        app,
        [
            "convert",
            str(make_epub()),
            "--tts-backend",
            "stub",
            "--format",
            "m4b",
            "--jobs-dir",
            str(jobs_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    m4bs = list(jobs_dir.glob("*/output/*.m4b"))
    assert len(m4bs) == 1
    # fixture EPUB: References chapter is back matter and excluded -> 2 chapters
    assert probe_chapters(m4bs[0]) == ["Chapter One", "Chapter Two"]


def test_jobs_lists_created_job(make_epub, tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    job = Job.create(jobs_dir, JobConfig(source="whatever.epub"))
    result = runner.invoke(app, ["jobs", "--jobs-dir", str(jobs_dir)])
    assert result.exit_code == 0
    assert job.state.id in result.output
    line = next(ln for ln in result.output.splitlines() if job.state.id in ln)
    assert "!" not in line  # nothing wrong with this job


def test_jobs_flags_a_job_carrying_errors(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    job = Job.create(jobs_dir, JobConfig(source="whatever.epub"))
    job.state.errors["synthesize:degraded"] = "001"
    job.save()

    result = runner.invoke(app, ["jobs", "--jobs-dir", str(jobs_dir)])
    assert result.exit_code == 0
    line = next(ln for ln in result.output.splitlines() if job.state.id in ln)
    assert "!" in line


def test_convert_rejects_missing_file(tmp_path: Path):
    result = runner.invoke(
        app, ["convert", str(tmp_path / "nope.epub"), "--jobs-dir", str(tmp_path / "jobs")]
    )
    assert result.exit_code != 0


def test_convert_reports_stage_failure_without_a_traceback(make_epub, tmp_path: Path, monkeypatch):
    # "exploding" passes preflight (registered and local) and fails mid-run, which is the
    # only way to reach the pipeline's error path now that unusable choices are rejected up front.
    monkeypatch.setattr(synth_stage, "_RETRY_BACKOFF_SECONDS", (0, 0))  # don't sleep the suite
    result = runner.invoke(
        app,
        [
            "convert",
            str(make_epub()),
            "--tts-backend",
            "exploding",
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ],
    )
    assert result.exit_code == 1
    assert "error: TTS produced no audio" in result.output  # the stage's own wording
    assert "Traceback" not in result.output


def test_convert_rejects_unimplemented_mode_before_creating_a_job(make_epub, tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    result = runner.invoke(
        app,
        ["convert", str(make_epub()), "--mode", "rewrite", "--jobs-dir", str(jobs_dir)],
    )
    assert result.exit_code != 0
    assert "verbatim" in result.output
    assert not jobs_dir.exists()  # failed before any work was started


def test_convert_rejects_kokoro_without_models(make_epub, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ABC_MODELS_DIR", str(tmp_path / "absent"))
    jobs_dir = tmp_path / "jobs"
    result = runner.invoke(
        app,
        ["convert", str(make_epub()), "--tts-backend", "kokoro", "--jobs-dir", str(jobs_dir)],
    )
    assert result.exit_code != 0
    assert "kokoro-onnx/releases" in result.output
    assert not jobs_dir.exists()


def test_convert_rejects_unknown_backend_before_creating_a_job(make_epub, tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    result = runner.invoke(
        app,
        ["convert", str(make_epub()), "--tts-backend", "nope", "--jobs-dir", str(jobs_dir)],
    )
    assert result.exit_code != 0
    assert "unknown TTS backend" in result.output
    assert not jobs_dir.exists()


def test_preview_reports_backend_failure_cleanly(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    job = Job.create(jobs_dir, JobConfig(source="x.epub", tts_backend="nope"))
    (job.processed_dir / "000.txt").write_text("Hello preview.", encoding="utf-8")

    result = runner.invoke(app, ["preview", job.state.id, "--jobs-dir", str(jobs_dir)])
    assert result.exit_code == 1
    assert "unknown TTS backend" in result.output
    assert "Traceback" not in result.output


def test_jobs_survives_one_corrupt_job(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    good = Job.create(jobs_dir, JobConfig(source="good.epub"))
    bad = Job.create(jobs_dir, JobConfig(source="bad.epub"))
    (bad.dir / "job.json").write_text("{not json at all", encoding="utf-8")

    result = runner.invoke(app, ["jobs", "--jobs-dir", str(jobs_dir)])
    assert result.exit_code == 0
    assert good.state.id in result.output  # listing completed despite the bad one
    bad_line = next(ln for ln in result.output.splitlines() if bad.state.id in ln)
    assert "corrupt" in bad_line


def test_resume_unknown_job_id_exits_cleanly(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    result = runner.invoke(app, ["resume", "deadbeef", "--jobs-dir", str(jobs_dir)])
    assert result.exit_code == 2
    assert "deadbeef" in result.output
    assert not isinstance(result.exception, FileNotFoundError)


def test_doctor_runs():
    result = runner.invoke(app, ["doctor"])
    assert "ffmpeg" in result.output.lower()


def test_preview_requires_processed_text(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    job = Job.create(jobs_dir, JobConfig(source="x.epub", tts_backend="stub"))
    result = runner.invoke(app, ["preview", job.state.id, "--jobs-dir", str(jobs_dir)])
    assert result.exit_code != 0  # no processed text yet

    (job.processed_dir / "000.txt").write_text("Hello preview world.", encoding="utf-8")
    result = runner.invoke(app, ["preview", job.state.id, "--jobs-dir", str(jobs_dir)])
    assert result.exit_code == 0, result.output
    assert (job.output_dir / "preview.wav").exists()
