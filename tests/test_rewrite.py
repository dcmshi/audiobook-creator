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
