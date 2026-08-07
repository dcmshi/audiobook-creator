from pathlib import Path

import pytest

from audiobook_creator.core import engine
from audiobook_creator.core.job import Job
from audiobook_creator.models import STAGES, JobConfig, StageStatus


def _fake_stages(calls: list[str], fail_at: str | None = None):
    def make(name):
        def stage(job):
            if name == fail_at:
                raise ValueError(f"boom in {name}")
            calls.append(name)
        return stage

    return {name: make(name) for name in STAGES}


def test_runs_all_stages_in_order(tmp_path: Path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(engine, "get_stages", lambda: _fake_stages(calls))
    job = Job.create(tmp_path, JobConfig(source="x.epub"))
    engine.run(job)
    assert calls == STAGES
    assert all(s is StageStatus.DONE for s in job.state.stages.values())


def test_skips_completed_stages(tmp_path: Path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(engine, "get_stages", lambda: _fake_stages(calls))
    job = Job.create(tmp_path, JobConfig(source="x.epub"))
    job.state.stages["ingest"] = StageStatus.DONE
    job.state.stages["structure"] = StageStatus.DONE
    engine.run(job)
    assert calls == ["process", "synthesize", "package"]


def test_failure_recorded_and_raised(tmp_path: Path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(engine, "get_stages", lambda: _fake_stages(calls, fail_at="process"))
    job = Job.create(tmp_path, JobConfig(source="x.epub"))
    with pytest.raises(ValueError):
        engine.run(job)
    assert job.state.stages["process"] is StageStatus.FAILED
    assert "boom in process" in job.state.errors["process"]
    # persisted to disk too
    reloaded = Job.load(tmp_path, job.state.id)
    assert reloaded.state.stages["process"] is StageStatus.FAILED


def test_from_stage_resets_later_stages(tmp_path: Path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(engine, "get_stages", lambda: _fake_stages(calls))
    job = Job.create(tmp_path, JobConfig(source="x.epub"))
    for name in STAGES:
        job.state.stages[name] = StageStatus.DONE
    engine.run(job, from_stage="synthesize")
    assert calls == ["synthesize", "package"]
