import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audiobook_creator.core.job import Job
from audiobook_creator.models import JobConfig
from audiobook_creator.web.app import create_app
from audiobook_creator.web.runner import JobRunner


class InlineRunner(JobRunner):
    """Executes submissions synchronously so tests are deterministic."""

    def submit(self, job_id, fn):
        fn()


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(tmp_path / "jobs", runner=InlineRunner())
    with TestClient(app) as c:
        yield c, tmp_path / "jobs"


def test_jobs_empty(client):
    c, _ = client
    assert c.get("/api/jobs").json() == []


def test_jobs_lists_existing(client):
    c, jobs_dir = client
    job = Job.create(jobs_dir, JobConfig(source="x.epub"))
    rows = c.get("/api/jobs").json()
    assert rows[0]["id"] == job.state.id
    assert rows[0]["stages"]["ingest"] == "pending"


def test_corrupt_job_row_does_not_abort(client):
    c, jobs_dir = client
    Job.create(jobs_dir, JobConfig(source="x.epub"))
    bad = jobs_dir / "deadbeef"
    bad.mkdir()
    (bad / "job.json").write_text("{not json", encoding="utf-8")
    rows = c.get("/api/jobs").json()
    assert len(rows) == 2
    assert any(r.get("corrupt") for r in rows)


def test_bad_job_id_is_404(client):
    c, _ = client
    assert c.get("/api/jobs/../etc").status_code == 404
    assert c.get("/api/jobs/zzzzzzzz").status_code == 404


def test_job_detail_reports_files_and_activity(client):
    c, jobs_dir = client
    job = Job.create(jobs_dir, JobConfig(source="x.epub"))
    (job.processed_dir / "000.txt").write_text("narration", encoding="utf-8")
    (job.output_dir / "book.m4b").write_bytes(b"audio")
    body = c.get(f"/api/jobs/{job.state.id}").json()
    assert body["id"] == job.state.id
    assert body["processed"] == ["000.txt"]
    assert body["outputs"] == ["book.m4b"]
    assert body["active"] is False


def test_listing_reports_output_presence(client):
    c, jobs_dir = client
    job = Job.create(jobs_dir, JobConfig(source="x.epub"))
    assert c.get("/api/jobs").json()[0]["has_output"] is False
    (job.output_dir / "book.m4b").write_bytes(b"audio")
    assert c.get("/api/jobs").json()[0]["has_output"] is True


def test_runner_shuts_down_with_the_app(tmp_path: Path):
    """The lifespan owns the pool: leaving the context must release the worker thread."""

    class SpyRunner(JobRunner):
        def __init__(self):
            super().__init__()
            self.shut_down = False

        def shutdown(self):
            self.shut_down = True
            super().shutdown()

    runner = SpyRunner()
    with TestClient(create_app(tmp_path / "jobs", runner=runner)):
        pass
    assert runner.shut_down is True


def test_runner_survives_a_failing_pipeline():
    """job.json carries the failure; a raising pipeline must not kill the worker thread."""
    runner = JobRunner()
    second_ran = threading.Event()
    try:
        runner.submit("deadbeef", lambda: (_ for _ in ()).throw(RuntimeError("stage exploded")))
        runner.submit("deadbee0", second_ran.set)
        assert second_ran.wait(timeout=5) is True  # the worker took the next job regardless
        assert runner.is_active("deadbeef") is False
    finally:
        runner.shutdown()


def _upload(c, epub_bytes: bytes, **form):
    files = {"file": ("book.epub", epub_bytes, "application/epub+zip")}
    data = {
        "mode": "verbatim",
        "tts_backend": "stub",
        "voice": "af_heart",
        "formats": "m4b",
        "local_only": "false",
        "use_llm": "false",
        **form,
    }
    return c.post("/api/jobs", files=files, data=data)


def test_upload_creates_and_runs_job(client, make_epub_bytes):
    c, jobs_dir = client
    resp = _upload(c, make_epub_bytes())
    assert resp.status_code == 202
    job_id = resp.json()["id"]
    detail = c.get(f"/api/jobs/{job_id}").json()
    # InlineRunner ran the pipeline synchronously before the response was returned.
    assert detail["stages"]["package"] in ("done", "failed")
    assert (jobs_dir / job_id / "source" / "book.epub").exists()


def test_preflight_failure_leaves_no_job(client, make_epub_bytes):
    c, jobs_dir = client
    resp = _upload(c, make_epub_bytes(), tts_backend="nope")
    assert resp.status_code == 400
    assert "unknown" in resp.json()["detail"].lower()
    assert list(jobs_dir.glob("*/job.json")) == []


def test_bad_format_rejected(client, make_epub_bytes):
    c, jobs_dir = client
    assert _upload(c, make_epub_bytes(), formats="wav").status_code == 400
    assert list(jobs_dir.glob("*/job.json")) == []


def test_bad_mode_rejected(client, make_epub_bytes):
    c, jobs_dir = client
    assert _upload(c, make_epub_bytes(), mode="interpretive-dance").status_code == 400
    assert list(jobs_dir.glob("*/job.json")) == []


def test_unknown_llm_provider_rejected(client, make_epub_bytes):
    c, jobs_dir = client
    resp = _upload(c, make_epub_bytes(), use_llm="true", llm_provider="gpt")
    assert resp.status_code == 400
    assert list(jobs_dir.glob("*/job.json")) == []


def test_llm_provider_is_persisted(client, make_epub_bytes, monkeypatch):
    """Plan 2 gave jobs a provider; the upload form must not silently drop it."""
    from audiobook_creator.process import llm as llm_pkg

    class _Kimi:
        name = "kimi"

    monkeypatch.setattr(llm_pkg, "resolve_llm", lambda **kw: _Kimi())
    c, jobs_dir = client
    resp = _upload(c, make_epub_bytes(), use_llm="true", llm_provider="kimi")
    assert resp.status_code == 202
    job = Job.load(jobs_dir, resp.json()["id"])
    assert job.state.config.llm_provider == "kimi"


def test_upload_source_is_recorded_as_the_job_source(client, make_epub_bytes):
    c, jobs_dir = client
    job_id = _upload(c, make_epub_bytes()).json()["id"]
    job = Job.load(jobs_dir, job_id)
    assert Path(job.state.config.source).name == "book.epub"
    assert Path(job.state.config.source).is_file()


def test_hostile_upload_filename_is_contained(client, make_epub_bytes):
    """The filename is attacker-controlled; it must not escape the job's source directory."""
    c, jobs_dir = client
    files = {"file": ("../../evil.epub", make_epub_bytes(), "application/epub+zip")}
    data = {"mode": "verbatim", "tts_backend": "stub", "formats": "m4b", "use_llm": "false"}
    resp = c.post("/api/jobs", files=files, data=data)
    assert resp.status_code == 202
    job = Job.load(jobs_dir, resp.json()["id"])
    source = Path(job.state.config.source).resolve()
    assert source.is_relative_to((jobs_dir / job.state.id).resolve())
    assert not (jobs_dir.parent / "evil.epub").exists()


def test_dot_only_upload_filename_does_not_target_the_directory(client, make_epub_bytes):
    """safe_filename keeps dots, so ".." would resolve onto the source directory itself."""
    c, jobs_dir = client
    files = {"file": ("..", make_epub_bytes(), "application/epub+zip")}
    data = {"mode": "verbatim", "tts_backend": "stub", "formats": "m4b", "use_llm": "false"}
    resp = c.post("/api/jobs", files=files, data=data)
    assert resp.status_code == 202
    job = Job.load(jobs_dir, resp.json()["id"])
    assert Path(job.state.config.source).name == "upload.bin"
    assert Path(job.state.config.source).is_file()
