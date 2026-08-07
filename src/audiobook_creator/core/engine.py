from collections.abc import Callable

from audiobook_creator.core.job import Job
from audiobook_creator.models import STAGES, StageStatus

StageFunc = Callable[[Job], None]


def get_stages() -> dict[str, StageFunc]:
    """Lazy stage lookup so heavy deps import only when their stage runs."""
    from audiobook_creator.ingest.stage import run_stage as ingest
    from audiobook_creator.package.stage import run_stage as package
    from audiobook_creator.process.stage import run_stage as process
    from audiobook_creator.structure.stage import run_stage as structure
    from audiobook_creator.synthesize.stage import run_stage as synthesize

    return {
        "ingest": ingest,
        "structure": structure,
        "process": process,
        "synthesize": synthesize,
        "package": package,
    }


def run(job: Job, from_stage: str | None = None) -> None:
    if from_stage is not None:
        if from_stage not in STAGES:
            raise ValueError(f"unknown stage {from_stage!r}; expected one of {STAGES}")
        for name in STAGES[STAGES.index(from_stage):]:
            job.state.stages[name] = StageStatus.PENDING
        job.save()

    stages = get_stages()
    for name in STAGES:
        if job.state.stages[name] is StageStatus.DONE:
            continue
        job.state.stages[name] = StageStatus.RUNNING
        job.state.errors.pop(name, None)
        job.save()
        try:
            stages[name](job)
        except Exception as exc:
            job.state.stages[name] = StageStatus.FAILED
            job.state.errors[name] = f"{type(exc).__name__}: {exc}"
            job.save()
            raise
        job.state.stages[name] = StageStatus.DONE
        job.save()
