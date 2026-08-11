import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class JobRunner:
    """Runs pipelines off the request thread, one at a time process-wide.

    A single worker is the point, not a limitation: the stages are CPU- and disk-heavy, and
    two concurrent runs over the same jobs directory would compete for the same files.
    """

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="abc-pipeline")
        self._active: set[str] = set()

    def submit(self, job_id: str, fn: Callable[[], None]) -> None:
        self._active.add(job_id)

        def wrapped() -> None:
            try:
                fn()
            except Exception:
                # Swallowed on purpose: the engine has already persisted FAILED to job.json,
                # which is what the UI reads. Letting this escape would kill the only worker
                # and strand every job queued behind it.
                logger.exception("pipeline run for job %s failed", job_id)
            finally:
                self._active.discard(job_id)

        self._pool.submit(wrapped)

    def is_active(self, job_id: str) -> bool:
        return job_id in self._active

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
