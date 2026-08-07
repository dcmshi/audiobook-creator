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
from audiobook_creator.structure.chapters import classify_matter, split_chapters
from audiobook_creator.structure.stage import run_stage


def _doc(blocks: list[Block]) -> Document:
    return Document(meta=DocumentMeta(title="T"), blocks=blocks)


def _h(text: str, level: int = 1) -> Block:
    return Block(type=BlockType.HEADING, text=text, level=level)


def _p(text: str) -> Block:
    return Block(type=BlockType.PARAGRAPH, text=text)


def test_split_on_level1_headings():
    doc = _doc([_h("One"), _p("a"), _h("Two"), _p("b")])
    chapters = split_chapters(doc)
    assert [c.title for c in chapters] == ["One", "Two"]
    assert chapters[0].index == 0
    assert chapters[1].blocks[0].text == "b"


def test_leading_blocks_become_beginning_chapter():
    doc = _doc([_p("preamble"), _h("One"), _p("a")])
    chapters = split_chapters(doc)
    assert chapters[0].title == "Beginning"
    assert chapters[0].blocks[0].text == "preamble"


def test_fallback_to_level2_when_single_level1():
    doc = _doc([_h("Paper Title"), _h("Intro", 2), _p("a"), _h("Methods", 2), _p("b")])
    chapters = split_chapters(doc)
    titles = [c.title for c in chapters]
    assert "Intro" in titles and "Methods" in titles


def test_classify_matter_keywords():
    assert classify_matter("References") is Matter.BACK
    assert classify_matter("Bibliography") is Matter.BACK
    assert classify_matter("Index") is Matter.BACK
    assert classify_matter("Table of Contents") is Matter.FRONT
    assert classify_matter("Copyright") is Matter.FRONT
    assert classify_matter("Preface") is Matter.FRONT
    assert classify_matter("Chapter One") is Matter.BODY
    assert classify_matter("Some Odd Title") is Matter.BODY  # ambiguous -> body


def test_run_stage_writes_chapter_files(tmp_path: Path):
    job = Job.create(tmp_path, JobConfig(source="x.epub"))
    doc = _doc([_h("One"), _p("a"), _h("References"), _p("Doe 2026")])
    job.document_path.write_text(doc.model_dump_json(), encoding="utf-8")
    run_stage(job)
    files = sorted(job.chapters_dir.glob("*.json"))
    assert [f.name for f in files] == ["000.json", "001.json"]
    ch1 = Chapter.model_validate_json(files[1].read_text(encoding="utf-8"))
    assert ch1.matter is Matter.BACK
