from pathlib import Path

from helpers import requires_ffmpeg
from typer.testing import CliRunner

from audiobook_creator.cli import app
from audiobook_creator.core.job import Job
from audiobook_creator.models import JobConfig
from audiobook_creator.package.ffmpeg import probe_chapters

runner = CliRunner()


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
