from pathlib import Path

from audiobook_creator.core.job import Job
from audiobook_creator.models import (
    Block,
    BlockType,
    Chapter,
    Document,
    DocumentMeta,
    JobConfig,
    Matter,
)
from audiobook_creator.process.stage import run_stage
from audiobook_creator.process.verbatim import render_chapter_text
from audiobook_creator.structure.chapters import split_chapters


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
    assert text.startswith("One. [[pause]]")  # chapter title narrated first
    assert "Chapter One. [[pause]]" in text  # heading blocks still rendered
    assert "40 percent intensity." in text
    assert "Year 2026" not in text  # tables skipped in verbatim v1
    assert "ignore me" not in text  # footnotes skipped
    assert "Figure 1: A storm." in text  # captions kept


def test_split_chapter_title_is_narrated():
    # split_chapters consumes the boundary heading into Chapter.title, so the title
    # reaches the listener here or not at all.
    doc = Document(
        meta=DocumentMeta(title="T"),
        blocks=[
            Block(type=BlockType.HEADING, text="Chapter One", level=1),
            Block(type=BlockType.PARAGRAPH, text="It was a dark and stormy night."),
        ],
    )
    chapter = split_chapters(doc)[0]
    assert chapter.title == "Chapter One"
    assert not any(b.type is BlockType.HEADING for b in chapter.blocks)

    text = render_chapter_text(chapter)
    assert text.startswith("Chapter One. [[pause]]")
    assert "It was a dark and stormy night." in text


def test_beginning_placeholder_never_narrated():
    doc = Document(
        meta=DocumentMeta(title="T"),
        blocks=[
            Block(type=BlockType.PARAGRAPH, text="Front prose."),
            Block(type=BlockType.HEADING, text="Chapter One", level=1),
            Block(type=BlockType.PARAGRAPH, text="Body prose."),
        ],
    )
    beginning = split_chapters(doc)[0]
    assert beginning.title == "Beginning"

    text = render_chapter_text(beginning)
    assert "Beginning" not in text
    assert text.startswith("Front prose.")


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
