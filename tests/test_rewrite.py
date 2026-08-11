import logging
from pathlib import Path

import pytest

from audiobook_creator.core.job import Job
from audiobook_creator.models import Block, BlockType, Chapter, JobConfig, Mode
from audiobook_creator.process import llm as llm_pkg
from audiobook_creator.process import stage as process_stage
from audiobook_creator.process.llm.base import LLMError, LLMUnsupported
from audiobook_creator.process.rewrite import render_rewrite


class ScriptedLLM:
    name = "fake"
    model = "m"

    def __init__(self, vision_ok=True):
        self.vision_ok = vision_ok
        self.window_inputs = []

    def complete(self, user, *, system=None, max_tokens=2048):
        self.window_inputs.append(user)
        return f"REWRITTEN({len(user)} chars)"

    def describe_image(self, image_path, prompt, *, max_tokens=1024):
        if not self.vision_ok:
            raise LLMUnsupported("no vision")
        return "a rising trend line from 2020 to 2026"


def _chapter(tmp_path):
    fig = tmp_path / "fig-000.png"
    fig.write_bytes(b"png")
    return Chapter(
        index=0,
        title="Results",
        blocks=[
            Block(type=BlockType.PARAGRAPH, text="We measured growth."),
            Block(type=BlockType.TABLE, text="Year, Growth. 2020, 5. 2026, 40."),
            Block(type=BlockType.FIGURE, text="Figure 1: growth curve.", image_path=str(fig)),
        ],
    )


def test_rewrite_includes_title_table_and_vision(tmp_path: Path):
    llm = ScriptedLLM()
    out = render_rewrite(_chapter(tmp_path), llm, tmp_path / "cache")
    assert out.startswith("Results. [[pause]]")
    window = llm.window_inputs[0]
    assert "[TABLE]" in window
    assert "rising trend line" in window  # vision description reached the window


def test_vision_unsupported_falls_back_to_caption(tmp_path: Path):
    llm = ScriptedLLM(vision_ok=False)
    render_rewrite(_chapter(tmp_path), llm, tmp_path / "cache")
    assert "Figure 1: growth curve." in llm.window_inputs[0]


def test_window_failure_falls_back_to_verbatim_text(tmp_path: Path):
    class FailingLLM(ScriptedLLM):
        def complete(self, user, *, system=None, max_tokens=2048):
            raise LLMError("down")

    out = render_rewrite(_chapter(tmp_path), FailingLLM(), tmp_path / "cache")
    assert "We measured growth." in out  # content survived via fallback


def test_stage_dispatches_rewrite_and_records_backend(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        llm_pkg, "resolve_llm", lambda *, local_only, use_llm, provider=None: ScriptedLLM()
    )
    job = Job.create(tmp_path, JobConfig(source="x.epub", mode=Mode.REWRITE))
    ch = _chapter(tmp_path)
    (job.chapters_dir / "000.json").write_text(ch.model_dump_json(), encoding="utf-8")
    process_stage.run_stage(job)
    assert "llm:fake" in job.state.backends_used
    assert "REWRITTEN" in (job.processed_dir / "000.txt").read_text(encoding="utf-8")


def test_stage_refuses_rewrite_without_a_client(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        llm_pkg, "resolve_llm", lambda *, local_only, use_llm, provider=None: None
    )
    job = Job.create(tmp_path, JobConfig(source="x.epub", mode=Mode.REWRITE))
    ch = _chapter(tmp_path)
    (job.chapters_dir / "000.json").write_text(ch.model_dump_json(), encoding="utf-8")
    with pytest.raises(RuntimeError, match="--llm"):
        process_stage.run_stage(job)


class EchoingLLM(ScriptedLLM):
    """Returns prose that leaks input-side markers and un-verbalized symbols."""

    def complete(self, user, *, system=None, max_tokens=2048):
        return (
            "[TABLE] Growth reached 40% by 2026. [[pause]] "
            "[FIGURE fig-000] The figure shows a rise.\n\n"
            "A second paragraph [[TABLE]] with strays ]] and Fig. 2 nearby."
        )


def test_pause_survives_the_rules_pass_and_symbols_are_spoken(tmp_path: Path):
    out = render_rewrite(_chapter(tmp_path), EchoingLLM(), tmp_path / "cache")
    assert "[[pause]]" in out  # the one marker the prompt allows must survive normalize
    assert "40 percent" in out  # rules caught what the model left as "40%"
    assert "Figure 2" in out  # and the rest of the rule set applies too


def test_leaked_markers_are_stripped(tmp_path: Path):
    out = render_rewrite(_chapter(tmp_path), EchoingLLM(), tmp_path / "cache")
    body = out.split("[[pause]]", 1)[1]  # drop the deterministic title frame
    assert "[TABLE]" not in body
    assert "[FIGURE" not in body
    assert "]]" not in body.replace("[[pause]]", "")
    assert "[[" not in body.replace("[[pause]]", "")


def test_paragraph_breaks_survive_the_rules_pass(tmp_path: Path):
    out = render_rewrite(_chapter(tmp_path), EchoingLLM(), tmp_path / "cache")
    assert "A second paragraph" in out
    assert "\n\nA second paragraph" in out  # normalize must not weld paragraphs together


def test_stage_accepts_table_only_chapter_in_rewrite(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        llm_pkg, "resolve_llm", lambda *, local_only, use_llm, provider=None: ScriptedLLM()
    )
    job = Job.create(tmp_path, JobConfig(source="x.epub", mode=Mode.REWRITE))
    ch = Chapter(
        index=0,
        title="Data",
        blocks=[Block(type=BlockType.TABLE, text="Year, Growth. 2026, 40.")],
    )
    (job.chapters_dir / "000.json").write_text(ch.model_dump_json(), encoding="utf-8")
    process_stage.run_stage(job)  # must not raise: the table is what rewrite verbalizes
    assert (job.processed_dir / "000.txt").read_text(encoding="utf-8").strip()


class MarkdownLLM(ScriptedLLM):
    def complete(self, user, *, system=None, max_tokens=2048):
        return "## Results heading\n\n**Growth** was strong, see <em>the table</em>."


def test_markdown_output_falls_back_to_verbatim(tmp_path: Path):
    out = render_rewrite(_chapter(tmp_path), MarkdownLLM(), tmp_path / "cache")
    assert "#" not in out
    assert "**" not in out
    assert "<" not in out
    assert "We measured growth." in out  # content survived through the verbatim fallback


def test_footnotes_do_not_reach_the_window(tmp_path: Path):
    llm = ScriptedLLM()
    chapter = _chapter(tmp_path)
    chapter.blocks.append(Block(type=BlockType.FOOTNOTE, text="1. See the appendix for detail."))
    render_rewrite(chapter, llm, tmp_path / "cache")
    assert "See the appendix" not in llm.window_inputs[0]


def test_unexpected_vision_error_falls_back_to_caption(tmp_path: Path):
    class OddVisionLLM(ScriptedLLM):
        def describe_image(self, image_path, prompt, *, max_tokens=1024):
            raise ValueError("provider returned something unexpected")

    llm = OddVisionLLM()
    render_rewrite(_chapter(tmp_path), llm, tmp_path / "cache")  # must not raise
    assert "Figure 1: growth curve." in llm.window_inputs[0]


class MarkupVisionLLM(ScriptedLLM):
    """Vision output carries markup, and the window rewrite is rejected — so the fallback runs."""

    def complete(self, user, *, system=None, max_tokens=2048):
        return "## Rejected heading"

    def describe_image(self, image_path, prompt, *, max_tokens=1024):
        return "**The figure shows** a rise, see <em>panel A</em>."


def test_fallback_text_is_marker_free(tmp_path: Path):
    out = render_rewrite(_chapter(tmp_path), MarkupVisionLLM(), tmp_path / "cache")
    assert "**" not in out
    assert "<" not in out
    assert "#" not in out
    assert "The figure shows" in out  # the description survived, minus its markup
    assert "We measured growth." in out


def test_empty_rewrite_logs_and_falls_back(tmp_path: Path, caplog):
    class EmptyLLM(ScriptedLLM):
        def complete(self, user, *, system=None, max_tokens=2048):
            return "   "

    with caplog.at_level(logging.WARNING):
        out = render_rewrite(_chapter(tmp_path), EmptyLLM(), tmp_path / "cache")
    assert "no text" in caplog.text  # the third road warns like the other two
    assert "We measured growth." in out


class HeadingVisionLLM(ScriptedLLM):
    """Rewrite gets rejected, so the fallback runs over a description with a heading in it."""

    def complete(self, user, *, system=None, max_tokens=2048):
        return "## Rejected heading"

    def describe_image(self, image_path, prompt, *, max_tokens=1024):
        return "First part.\n\n## Second part heading"


def test_fallback_preserves_comparison_operators_across_blocks(tmp_path: Path):
    """A stray < in one block and > in a later one must not delete everything between."""
    chapter = Chapter(
        index=0,
        title="Math",
        blocks=[
            Block(type=BlockType.PARAGRAPH, text="The guard holds if a < b in every case."),
            Block(type=BlockType.PARAGRAPH, text="It fails when c > d, which we never allow."),
        ],
    )
    out = render_rewrite(chapter, MarkupVisionLLM(), tmp_path / "cache")
    assert "if a < b in every case" in out
    assert "It fails when c > d" in out


def test_fallback_keeps_paragraph_break_before_heading(tmp_path: Path):
    out = render_rewrite(_chapter(tmp_path), HeadingVisionLLM(), tmp_path / "cache")
    assert "First part.\n\nSecond part heading" in out


def test_windows_group_blocks_under_the_char_limit():
    """Pins the grouping that keeps each request inside the window budget."""
    from audiobook_creator.process.rewrite import _windows

    assert _windows(["x" * 2500, "y" * 2500, "z" * 2500, "w" * 100]) == [[0, 1], [2, 3]]
    assert _windows(["q" * 9000]) == [[0]]  # one oversized block stays whole
    assert _windows([]) == []


def test_vision_cache_is_keyed_on_image_bytes_not_path(tmp_path: Path):
    """A spend control: re-ingesting a book renumbers fig-NNN, and descriptions cost money."""
    calls: list[Path] = []

    class CountingVisionLLM(ScriptedLLM):
        def describe_image(self, image_path, prompt, *, max_tokens=1024):
            calls.append(image_path)
            return "a chart showing a rise"

    llm = CountingVisionLLM()
    for name in ("fig-000.png", "fig-007.png", "fig-000.png"):
        path = tmp_path / name
        path.write_bytes(b"IDENTICAL-BYTES")
        chapter = Chapter(
            index=0,
            title="T",
            blocks=[Block(type=BlockType.FIGURE, text="caption", image_path=str(path))],
        )
        render_rewrite(chapter, llm, tmp_path / "cache")
    assert len(calls) == 1
