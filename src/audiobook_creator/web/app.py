import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from audiobook_creator.core.job import Job
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
