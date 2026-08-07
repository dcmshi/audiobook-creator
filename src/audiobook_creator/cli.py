import shutil
from pathlib import Path

import typer

from audiobook_creator.core import engine
from audiobook_creator.core.job import Job
from audiobook_creator.models import JobConfig, Mode, StageStatus

app = typer.Typer(help="Turn PDFs and EPUBs into audiobooks.", no_args_is_help=True)

_STATUS_ICON = {
    StageStatus.PENDING: "·",
    StageStatus.RUNNING: "~",
    StageStatus.DONE: "+",
    StageStatus.FAILED: "x",
}


def _load_job(jobs_dir: Path, job_id: str) -> Job:
    try:
        return Job.load(jobs_dir, job_id)
    except FileNotFoundError:
        typer.echo(f"error: no job {job_id!r} under {jobs_dir}", err=True)
        raise typer.Exit(code=2) from None


@app.command()
def convert(
    source: str = typer.Argument(..., help="Path to EPUB/PDF/DOCX or URL"),
    mode: Mode = typer.Option(Mode.VERBATIM, "--mode"),
    tts_backend: str = typer.Option("kokoro", "--tts-backend"),
    voice: str = typer.Option("af_heart", "--voice"),
    formats: list[str] = typer.Option(["m4b"], "--format", help="m4b and/or mp3"),
    local_only: bool = typer.Option(
        False, "--local-only", help="Hard-block all network backends for this job"
    ),
    jobs_dir: Path = typer.Option(Path("jobs"), "--jobs-dir"),
) -> None:
    if not source.startswith(("http://", "https://")) and not Path(source).is_file():
        typer.echo(f"error: source file not found: {source}", err=True)
        raise typer.Exit(code=2)
    for fmt in formats:
        if fmt not in ("mp3", "m4b"):
            typer.echo(f"error: unknown format {fmt!r} (use mp3 or m4b)", err=True)
            raise typer.Exit(code=2)
    config = JobConfig(
        source=source,
        mode=mode,
        tts_backend=tts_backend,
        voice=voice,
        local_only=local_only,
        formats=formats,
    )
    job = Job.create(jobs_dir, config)
    typer.echo(f"job {job.state.id}: {source}")
    engine.run(job)
    for out in sorted(job.output_dir.iterdir()):
        if out.suffix in (".m4b", ".mp3"):
            typer.echo(f"  -> {out}")


@app.command()
def resume(
    job_id: str,
    jobs_dir: Path = typer.Option(Path("jobs"), "--jobs-dir"),
    from_stage: str = typer.Option(
        None, "--from-stage", help="Reset this stage and later ones, then run"
    ),
) -> None:
    job = _load_job(jobs_dir, job_id)
    engine.run(job, from_stage=from_stage)
    typer.echo(f"job {job_id} complete")


@app.command()
def jobs(jobs_dir: Path = typer.Option(Path("jobs"), "--jobs-dir")) -> None:
    ids = Job.list_ids(jobs_dir)
    if not ids:
        typer.echo("no jobs")
        return
    for job_id in ids:
        job = _load_job(jobs_dir, job_id)
        stages = " ".join(
            f"{name}:{_STATUS_ICON[status]}" for name, status in job.state.stages.items()
        )
        # Errors can outlive a successful run: the synthesize stage records degraded
        # output under compound keys the engine does not clear.
        warning = "!" if job.state.errors else " "
        typer.echo(
            f"{job_id} {warning} {job.state.config.mode.value:<8}  {stages}  "
            f"{job.state.config.source}"
        )


@app.command()
def doctor() -> None:
    problems = 0

    ffmpeg = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
    typer.echo(f"ffmpeg/ffprobe: {'OK' if ffmpeg else 'MISSING - install ffmpeg, add to PATH'}")
    if not ffmpeg:
        problems += 1

    try:
        import docling  # noqa: F401

        typer.echo("docling (PDF support): OK")
    except ImportError:
        typer.echo("docling (PDF support): not installed - EPUB only. Install: uv sync --extra pdf")

    from audiobook_creator.synthesize.kokoro import model_files_status

    typer.echo(f"kokoro models: {model_files_status()}")

    raise typer.Exit(code=1 if problems else 0)


if __name__ == "__main__":
    app()
