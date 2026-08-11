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


def test_classify_matter_does_not_swallow_body_titles():
    # A body chapter misfiled as front/back matter is deleted by the process stage's
    # body-only filter, which is the one failure direction the spec forbids.
    for title in (
        "Notes from the Field",
        "Notes on a Scandal",
        "Index Funds Explained",
        "Indexing Strategies",
        "Copyright Law in Practice",
        "Dedication and Discipline",
        "Prefaces I Have Known",
        "References to Popular Culture",
    ):
        assert classify_matter(title) is Matter.BODY, title


def test_classify_matter_still_catches_jargon_prefixes():
    assert classify_matter("Appendix A: Data") is Matter.BACK
    assert classify_matter("Table of Contents") is Matter.FRONT


def test_run_stage_writes_chapter_files(tmp_path: Path):
    job = Job.create(tmp_path, JobConfig(source="x.epub"))
    doc = _doc([_h("One"), _p("a"), _h("References"), _p("Doe 2026")])
    job.document_path.write_text(doc.model_dump_json(), encoding="utf-8")
    run_stage(job)
    files = sorted(job.chapters_dir.glob("*.json"))
    assert [f.name for f in files] == ["000.json", "001.json"]
    ch1 = Chapter.model_validate_json(files[1].read_text(encoding="utf-8"))
    assert ch1.matter is Matter.BACK


class _TiebreakLLM:
    name = "fake"
    model = "m"

    def __init__(self, reply: str = "0: front\n1: body\n2: back\n3: back"):
        self.reply = reply

    def complete(self, user, *, system=None, max_tokens=2048):
        return self.reply

    def describe_image(self, p, q, *, max_tokens=1024):
        return ""


def _edge_chapters() -> list[Chapter]:
    return [
        Chapter(index=0, title="Copyright", matter=Matter.FRONT, blocks=[]),
        Chapter(index=1, title="Notes", matter=Matter.BACK, blocks=[_p("real content")]),
        Chapter(index=2, title="The Middle", blocks=[_p("x")]),
        Chapter(index=3, title="Sources", blocks=[_p("bibliography text")]),
    ]


def test_llm_tiebreaker_rescues_body_and_respects_edge_gate():
    from audiobook_creator.structure.chapters import refine_matter_with_llm

    out = refine_matter_with_llm(_edge_chapters(), _TiebreakLLM())
    assert out[1].matter is Matter.BODY  # rescued: to-BODY is always allowed
    assert out[2].matter is Matter.BODY  # mid-document flip to BACK rejected
    assert out[3].matter is Matter.BACK  # last-3 edge: flip accepted


def test_llm_tiebreaker_returns_input_unchanged_on_unparsable_reply():
    from audiobook_creator.structure.chapters import refine_matter_with_llm

    before = _edge_chapters()
    out = refine_matter_with_llm(before, _TiebreakLLM(reply="I could not classify these."))
    assert [c.matter for c in out] == [c.matter for c in before]


def test_llm_tiebreaker_survives_a_failing_client():
    from audiobook_creator.process.llm.base import LLMError
    from audiobook_creator.structure.chapters import refine_matter_with_llm

    class BrokenLLM(_TiebreakLLM):
        def complete(self, user, *, system=None, max_tokens=2048):
            raise LLMError("down")

    before = _edge_chapters()
    out = refine_matter_with_llm(before, BrokenLLM())
    assert [c.matter for c in out] == [c.matter for c in before]


def test_llm_tiebreaker_refuses_to_empty_the_body():
    """Flipping every chapter out of BODY would leave nothing to narrate."""
    from audiobook_creator.structure.chapters import refine_matter_with_llm

    before = [
        Chapter(index=0, title="Preface", blocks=[_p("a")]),
        Chapter(index=1, title="Middle", blocks=[_p("b")]),
        Chapter(index=2, title="Appendix", blocks=[_p("c")]),
    ]
    out = refine_matter_with_llm(before, _TiebreakLLM(reply="0: front\n1: front\n2: back"))
    assert [c.matter for c in out] == [Matter.BODY] * 3


def test_llm_tiebreaker_refuses_to_reclassify_a_single_chapter_document():
    from audiobook_creator.structure.chapters import refine_matter_with_llm

    out = refine_matter_with_llm(
        [Chapter(index=0, title="Everything", blocks=[_p("a")])],
        _TiebreakLLM(reply="0: front"),
    )
    assert out[0].matter is Matter.BODY


def test_llm_tiebreaker_still_applies_flips_that_leave_body_behind():
    from audiobook_creator.structure.chapters import refine_matter_with_llm

    before = [
        Chapter(index=0, title="Preface", blocks=[_p("a")]),
        Chapter(index=1, title="Middle", blocks=[_p("b")]),
        Chapter(index=2, title="Appendix", blocks=[_p("c")]),
    ]
    out = refine_matter_with_llm(before, _TiebreakLLM(reply="0: front\n2: back"))
    assert [c.matter for c in out] == [Matter.FRONT, Matter.BODY, Matter.BACK]
