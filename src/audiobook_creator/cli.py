import logging
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import typer

from audiobook_creator.core import engine
from audiobook_creator.core.job import Job
from audiobook_creator.models import JobConfig, Mode, StageStatus

logger = logging.getLogger(__name__)

app = typer.Typer(help="Turn PDFs and EPUBs into audiobooks.", no_args_is_help=True)

PREVIEW_CHARS = 300

_STATUS_ICON = {
    StageStatus.PENDING: "·",
    StageStatus.RUNNING: "~",
    StageStatus.DONE: "+",
    StageStatus.FAILED: "x",
}


@app.callback()
def _main() -> None:
    # Without this the stages' warnings — notably synthesize's "N of M chunks failed" —
    # have no handler and never reach the user.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def _load_job(jobs_dir: Path, job_id: str) -> Job:
    try:
        return Job.load(jobs_dir, job_id)
    except FileNotFoundError:
        typer.echo(f"error: no job {job_id!r} under {jobs_dir}", err=True)
        raise typer.Exit(code=2) from None


@contextmanager
def _clean_errors() -> Iterator[None]:
    """Report a failure as one line instead of a traceback panel.

    Stage and backend errors carry text written to be read ("TTS failed for chapter(s)
    001"); letting them propagate buries that text.
    """
    try:
        yield
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001 - any stage error is a user-facing message
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from None


def _run_pipeline(job: Job, from_stage: str | None = None) -> None:
    with _clean_errors():
        engine.run(job, from_stage=from_stage)


def _preflight_or_exit(config: JobConfig) -> None:
    """Thin wrapper over the shared preflight: print what it says, exit 2 if it refuses."""
    from audiobook_creator.core.preflight import PreflightError, preflight

    try:
        warnings = preflight(config)
    except PreflightError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from None
    for warning in warnings:
        typer.echo(f"warning: {warning}", err=True)


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
    llm: str = typer.Option(
        None, "--llm", help="anthropic | kimi | ollama (paid APIs are billed per token)"
    ),
    no_llm: bool = typer.Option(
        False, "--no-llm", help="Rule-based text only; verbatim mode only"
    ),
    jobs_dir: Path = typer.Option(Path("jobs"), "--jobs-dir"),
) -> None:
    config = JobConfig(
        source=source,
        mode=mode,
        tts_backend=tts_backend,
        voice=voice,
        local_only=local_only,
        use_llm=not no_llm,
        llm_provider=llm,
        formats=formats,
    )
    _preflight_or_exit(config)
    job = Job.create(jobs_dir, config)
    typer.echo(f"job {job.state.id}: {source}")
    _run_pipeline(job)
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
    _run_pipeline(job, from_stage=from_stage)
    typer.echo(f"job {job_id} complete")


@app.command()
def jobs(jobs_dir: Path = typer.Option(Path("jobs"), "--jobs-dir")) -> None:
    ids = Job.list_ids(jobs_dir)
    if not ids:
        typer.echo("no jobs")
        return
    for job_id in ids:
        try:
            job = Job.load(jobs_dir, job_id)
        except (OSError, ValueError) as exc:
            # One unreadable job must not hide every other job in the list.
            logger.debug("could not read job %s: %s", job_id, exc)
            typer.echo(f"{job_id}    corrupt job.json")
            continue
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
def preview(
    job_id: str,
    jobs_dir: Path = typer.Option(Path("jobs"), "--jobs-dir"),
) -> None:
    """Synthesize ~30s from the first processed chapter to output/preview.wav."""
    from audiobook_creator.synthesize.base import chunk_text, get_backend, write_wav

    job = _load_job(jobs_dir, job_id)
    processed = sorted(job.processed_dir.glob("*.txt"))
    if not processed:
        typer.echo("error: no processed text yet - run convert (or resume) first", err=True)
        raise typer.Exit(code=2)
    text = processed[0].read_text(encoding="utf-8").replace("[[pause]]", " ")[:PREVIEW_CHARS]
    cfg = job.state.config
    out = job.output_dir / "preview.wav"
    with _clean_errors():
        backend = get_backend(cfg.tts_backend, local_only=cfg.local_only)
        pcm = b"".join(backend.synthesize(c, cfg.voice) for c in chunk_text(text))
        write_wav(out, pcm, backend.sample_rate)
    typer.echo(f"  -> {out}")


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

    # None of these affects the exit code: every LLM provider is optional, and the rule-based
    # verbatim path works with none of them configured.
    import os

    from audiobook_creator.process.llm.base import LLMError
    from audiobook_creator.process.llm.ollama_client import OllamaClient, base_url

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    typer.echo(f"anthropic: {'OK (key found)' if anthropic_key else 'not configured'}")
    kimi_key = os.environ.get("MOONSHOT_API_KEY")
    typer.echo(f"kimi: {'OK (key found)' if kimi_key else 'not configured'}")
    try:
        OllamaClient()
        typer.echo(f"ollama: OK ({base_url()})")
    except LLMError:
        typer.echo(f"ollama: not reachable ({base_url()})")

    raise typer.Exit(code=1 if problems else 0)


if __name__ == "__main__":
    app()
