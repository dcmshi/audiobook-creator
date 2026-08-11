import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile

from audiobook_creator.core import engine
from audiobook_creator.core.job import Job
from audiobook_creator.core.preflight import PreflightError, preflight
from audiobook_creator.models import JobConfig, Mode
from audiobook_creator.package.ffmpeg import safe_filename
from audiobook_creator.web.runner import JobRunner

# Job ids are the 8 hex characters Job.create mints. Matching that shape here is what makes
# "../etc" a 404 rather than a path that reaches the filesystem.
_JOB_ID = re.compile(r"^[0-9a-f]{8}$")

_OUTPUT_SUFFIXES = (".m4b", ".mp3", ".wav")


def _load_or_404(jobs_dir: Path, job_id: str) -> Job:
    if not _JOB_ID.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="unknown job")
    try:
        return Job.load(jobs_dir, job_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="unknown job") from exc


def create_app(jobs_dir: Path, runner: JobRunner | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        app.state.runner.shutdown()

    app = FastAPI(title="audiobook-creator", lifespan=lifespan)
    app.state.jobs_dir = jobs_dir
    app.state.runner = runner or JobRunner()

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        rows: list[dict] = []
        for job_id in Job.list_ids(jobs_dir):
            try:
                job = Job.load(jobs_dir, job_id)
            except (OSError, ValueError):
                # One unreadable job.json must not cost the user the rest of the listing.
                rows.append({"id": job_id, "corrupt": True})
                continue
            rows.append(
                {
                    "id": job_id,
                    "source": job.state.config.source,
                    "mode": job.state.config.mode.value,
                    "stages": {name: status.value for name, status in job.state.stages.items()},
                    "errors": job.state.errors,
                    # any() over the matches, not over the generators: a bare generator is
                    # always truthy, which would report every job as finished.
                    "has_output": any(job.output_dir.glob("*.m4b"))
                    or any(job.output_dir.glob("*.mp3")),
                }
            )
        return rows

    @app.post("/api/jobs", status_code=202)
    def create_job(
        file: UploadFile,
        mode: str = Form("verbatim"),
        tts_backend: str = Form("kokoro"),
        voice: str = Form("af_heart"),
        formats: str = Form("m4b"),
        local_only: bool = Form(False),
        use_llm: bool = Form(True),
        llm_provider: str = Form(""),
    ) -> dict:
        try:
            job_mode = Mode(mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"unknown mode {mode!r}") from exc
        # Preflighted with the uploaded filename standing in for the source: the bytes are
        # still in the request, so check_source is off and nothing has touched disk yet.
        config = JobConfig(
            source=file.filename or "upload",
            mode=job_mode,
            tts_backend=tts_backend,
            voice=voice,
            local_only=local_only,
            use_llm=use_llm,
            llm_provider=llm_provider.strip() or None,
            formats=[f.strip() for f in formats.split(",") if f.strip()],
        )
        try:
            warnings = preflight(config, check_source=False)
        except PreflightError as exc:
            # Before Job.create on purpose: a refused job leaves no directory behind.
            raise HTTPException(status_code=400, detail=str(exc)) from None

        job = Job.create(jobs_dir, config)
        source_dir = job.dir / "source"
        source_dir.mkdir(exist_ok=True)
        # The filename is attacker-controlled. safe_filename drops separators but keeps dots,
        # so "..", "." and "" would still resolve onto the directory itself.
        name = Path(safe_filename(file.filename or "upload")).name
        if not name or set(name) <= {"."}:
            name = "upload.bin"
        dest = source_dir / name
        dest.write_bytes(file.file.read())
        job.state.config.source = str(dest)
        job.save()
        app.state.runner.submit(job.state.id, lambda: engine.run(job))
        return {"id": job.state.id, "warnings": warnings}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = _load_or_404(jobs_dir, job_id)
        return {
            **job.state.model_dump(mode="json"),
            "processed": sorted(p.name for p in job.processed_dir.glob("*.txt")),
            # glob rather than iterdir: a job whose output directory was removed by hand
            # should read as "no outputs", not raise.
            "outputs": sorted(
                p.name for p in job.output_dir.glob("*") if p.suffix in _OUTPUT_SUFFIXES
            ),
            "active": app.state.runner.is_active(job_id),
        }

    return app
