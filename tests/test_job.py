from pathlib import Path

import pytest

from audiobook_creator.core.job import Job
from audiobook_creator.models import JobConfig, StageStatus


def test_create_makes_directories_and_state(tmp_path: Path):
    job = Job.create(tmp_path, JobConfig(source="book.epub"))
    assert job.dir == tmp_path / job.state.id
    for d in (job.chapters_dir, job.processed_dir, job.audio_dir, job.output_dir, job.assets_dir):
        assert d.is_dir()
    assert (job.dir / "job.json").is_file()
    assert all(s is StageStatus.PENDING for s in job.state.stages.values())


def test_load_roundtrip(tmp_path: Path):
    created = Job.create(tmp_path, JobConfig(source="book.epub", voice="am_adam"))
    created.state.stages["ingest"] = StageStatus.DONE
    created.save()

    loaded = Job.load(tmp_path, created.state.id)
    assert loaded.state.config.voice == "am_adam"
    assert loaded.state.stages["ingest"] is StageStatus.DONE


def test_save_leaves_no_temp_file_behind(tmp_path: Path):
    job = Job.create(tmp_path, JobConfig(source="book.epub"))
    job.save()
    assert list(job.dir.glob("*.tmp")) == []


def test_failed_save_preserves_the_previous_state(tmp_path: Path, monkeypatch):
    job = Job.create(tmp_path, JobConfig(source="book.epub"))
    original = (job.dir / "job.json").read_text(encoding="utf-8")
    real_write_text = Path.write_text

    def truncate_then_fail(self: Path, *args, **kwargs):
        # What an interrupted write really does: the file is opened and emptied first.
        real_write_text(self, "", encoding="utf-8")
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", truncate_then_fail)
    job.state.stages["ingest"] = StageStatus.DONE
    with pytest.raises(OSError):
        job.save()
    monkeypatch.undo()

    # Writing in place would have emptied job.json; writing aside cannot touch it.
    assert (job.dir / "job.json").read_text(encoding="utf-8") == original
    assert list(job.dir.glob("*.tmp")) == []


def test_list_ids(tmp_path: Path):
    a = Job.create(tmp_path, JobConfig(source="a.epub"))
    b = Job.create(tmp_path, JobConfig(source="b.epub"))
    assert set(Job.list_ids(tmp_path)) == {a.state.id, b.state.id}
