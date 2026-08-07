import uuid
from pathlib import Path

from audiobook_creator.models import STAGES, JobConfig, JobState, StageStatus


class Job:
    def __init__(self, jobs_dir: Path, state: JobState):
        self.jobs_dir = jobs_dir
        self.state = state

    # --- paths ---
    @property
    def dir(self) -> Path:
        return self.jobs_dir / self.state.id

    @property
    def document_path(self) -> Path:
        return self.dir / "document.json"

    @property
    def chapters_dir(self) -> Path:
        return self.dir / "chapters"

    @property
    def processed_dir(self) -> Path:
        return self.dir / "processed"

    @property
    def audio_dir(self) -> Path:
        return self.dir / "audio"

    @property
    def output_dir(self) -> Path:
        return self.dir / "output"

    @property
    def assets_dir(self) -> Path:
        return self.dir / "assets"

    # --- lifecycle ---
    @classmethod
    def create(cls, jobs_dir: Path, config: JobConfig) -> "Job":
        state = JobState(
            id=uuid.uuid4().hex[:8],
            config=config,
            stages={name: StageStatus.PENDING for name in STAGES},
        )
        job = cls(jobs_dir, state)
        for d in (job.chapters_dir, job.processed_dir, job.audio_dir, job.output_dir, job.assets_dir):
            d.mkdir(parents=True, exist_ok=True)
        job.save()
        return job

    @classmethod
    def load(cls, jobs_dir: Path, job_id: str) -> "Job":
        raw = (jobs_dir / job_id / "job.json").read_text(encoding="utf-8")
        return cls(jobs_dir, JobState.model_validate_json(raw))

    @classmethod
    def list_ids(cls, jobs_dir: Path) -> list[str]:
        if not jobs_dir.is_dir():
            return []
        return sorted(p.parent.name for p in jobs_dir.glob("*/job.json"))

    def save(self) -> None:
        (self.dir / "job.json").write_text(self.state.model_dump_json(indent=2), encoding="utf-8")
