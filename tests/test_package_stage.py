import json
import subprocess
from pathlib import Path

from helpers import requires_ffmpeg

from audiobook_creator.core.job import Job
from audiobook_creator.models import (
    Block,
    BlockType,
    Chapter,
    Document,
    DocumentMeta,
    JobConfig,
)
from audiobook_creator.package.ffmpeg import safe_filename, write_ffmetadata
from audiobook_creator.package.stage import run_stage
from audiobook_creator.synthesize.base import write_wav


def test_safe_filename():
    assert safe_filename('Ch: "One" <draft?>') == "Ch One draft"


def test_ffmetadata_chapter_offsets(tmp_path: Path):
    meta = tmp_path / "meta.txt"
    write_ffmetadata(meta, title="Book", artist="Jane", chapters=[("One", 2.0), ("Two", 3.5)])
    content = meta.read_text(encoding="utf-8")
    assert content.startswith(";FFMETADATA1")
    assert "title=Book" in content and "artist=Jane" in content
    assert "START=0" in content and "END=2000" in content
    assert "START=2000" in content and "END=5500" in content
    assert "title=One" in content and "title=Two" in content


def _prepared_job(tmp_path: Path, formats: list[str]) -> Job:
    job = Job.create(tmp_path, JobConfig(source="x.epub", formats=formats))
    doc = Document(
        meta=DocumentMeta(title="Test Book", author="Jane Doe"),
        blocks=[Block(type=BlockType.PARAGRAPH, text="x")],
    )
    job.document_path.write_text(doc.model_dump_json(), encoding="utf-8")
    for i, title in enumerate(["One", "Two"]):
        ch = Chapter(index=i, title=title, blocks=[])
        (job.chapters_dir / f"{i:03d}.json").write_text(ch.model_dump_json(), encoding="utf-8")
        write_wav(job.audio_dir / f"{i:03d}.wav", b"\x00\x01" * 24000, 24000)  # 1s each
    return job


@requires_ffmpeg
def test_m4b_with_chapters(tmp_path: Path):
    job = _prepared_job(tmp_path, ["m4b"])
    run_stage(job)
    m4b = job.output_dir / "Test Book.m4b"
    assert m4b.exists()
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_chapters", str(m4b)],
        check=True,
        capture_output=True,
        text=True,
    )
    chapters = json.loads(probe.stdout)["chapters"]
    assert [c["tags"]["title"] for c in chapters] == ["One", "Two"]


@requires_ffmpeg
def test_mp3_per_chapter(tmp_path: Path):
    job = _prepared_job(tmp_path, ["mp3"])
    run_stage(job)
    mp3s = sorted(job.output_dir.glob("*.mp3"))
    assert [m.name for m in mp3s] == ["000 - One.mp3", "001 - Two.mp3"]


@requires_ffmpeg
def test_track_numbers_are_positional_not_derived_from_filename(tmp_path: Path):
    # Sparse numbering: front matter skipped, so the body starts at 001.
    job = Job.create(tmp_path, JobConfig(source="x.epub", formats=["mp3"]))
    doc = Document(
        meta=DocumentMeta(title="Test Book", author="Jane Doe"),
        blocks=[Block(type=BlockType.PARAGRAPH, text="x")],
    )
    job.document_path.write_text(doc.model_dump_json(), encoding="utf-8")
    for i, title in ((1, "One"), (3, "Two")):
        ch = Chapter(index=i, title=title, blocks=[])
        (job.chapters_dir / f"{i:03d}.json").write_text(ch.model_dump_json(), encoding="utf-8")
        write_wav(job.audio_dir / f"{i:03d}.wav", b"\x00\x01" * 24000, 24000)
    run_stage(job)

    assert [m.name for m in sorted(job.output_dir.glob("*.mp3"))] == [
        "001 - One.mp3",
        "003 - Two.mp3",
    ]
    tracks = [_track_number(m) for m in sorted(job.output_dir.glob("*.mp3"))]
    assert tracks == ["1", "2"]  # positional, not 2 and 4


def _track_number(mp3: Path) -> str:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            str(mp3),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["format"]["tags"]["track"]


@requires_ffmpeg
def test_m4b_builds_when_jobs_dir_is_relative(tmp_path: Path, monkeypatch):
    # Reproduces real CLI use: the default --jobs-dir is the RELATIVE path
    # "jobs", and ffmpeg resolves concat entries against the list file's dir.
    monkeypatch.chdir(tmp_path)
    job = _prepared_job(Path("jobs"), ["m4b"])
    run_stage(job)
    assert (job.output_dir / "Test Book.m4b").exists()
