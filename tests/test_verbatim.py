from pathlib import Path

from audiobook_creator.core.job import Job
from audiobook_creator.models import (
    Block,
    BlockType,
    Chapter,
    JobConfig,
    Matter,
)
from audiobook_creator.process.stage import run_stage
from audiobook_creator.process.verbatim import render_chapter_text


def test_render_heading_pause_and_normalized_paragraphs():
    ch = Chapter(
        index=0,
        title="One",
        blocks=[
            Block(type=BlockType.HEADING, text="Chapter One", level=1),
            Block(type=BlockType.PARAGRAPH, text="Rain fell at 40% intensity [3]."),
            Block(type=BlockType.TABLE, text="Year 2026"),
            Block(type=BlockType.FOOTNOTE, text="1. ignore me"),
            Block(type=BlockType.CAPTION, text="Figure 1: A storm."),
        ],
    )
    text = render_chapter_text(ch)
    assert text.startswith("Chapter One. [[pause]]")
    assert "40 percent intensity." in text
    assert "Year 2026" not in text  # tables skipped in verbatim v1
    assert "ignore me" not in text  # footnotes skipped
    assert "Figure 1: A storm." in text  # captions kept


def test_run_stage_writes_body_chapters_only(tmp_path: Path):
    job = Job.create(tmp_path, JobConfig(source="x.epub"))
    body = Chapter(
        index=0, title="One",
        blocks=[Block(type=BlockType.PARAGRAPH, text="Hello there.")],
    )
    back = Chapter(
        index=1, title="References", matter=Matter.BACK,
        blocks=[Block(type=BlockType.PARAGRAPH, text="Doe 2026.")],
    )
    (job.chapters_dir / "000.json").write_text(body.model_dump_json(), encoding="utf-8")
    (job.chapters_dir / "001.json").write_text(back.model_dump_json(), encoding="utf-8")
    run_stage(job)
    files = sorted(job.processed_dir.glob("*.txt"))
    assert [f.name for f in files] == ["000.txt"]
    assert "Hello there." in files[0].read_text(encoding="utf-8")
