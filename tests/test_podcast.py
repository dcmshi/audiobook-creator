import logging
from pathlib import Path

import pytest

from audiobook_creator.core.job import Job
from audiobook_creator.models import (
    Block,
    BlockType,
    Chapter,
    Document,
    DocumentMeta,
    JobConfig,
    Mode,
)
from audiobook_creator.process import llm as llm_pkg
from audiobook_creator.process import stage as process_stage
from audiobook_creator.process.llm.base import LLMError
from audiobook_creator.process.podcast import render_podcast

_SCRIPT = "\n".join(
    [
        "[[speaker:1]] Welcome, today we discuss storms.",
        "[[speaker:2]] Storms are loud, and the data proves it.",
        "[[speaker:1]] What did chapter two find?",
        "[[speaker:2]] Rainfall hit four hundred millimeters.",
    ]
)


class PodcastLLM:
    name = "fake"
    model = "m"

    def __init__(self):
        self.prompts = []

    def complete(self, user, *, system=None, max_tokens=2048):
        self.prompts.append(user)
        assert "Chapter One" in user  # source content reached the prompt
        return _SCRIPT

    def describe_image(self, p, q, *, max_tokens=1024):
        return ""


def _chapters():
    return [
        Chapter(
            index=0,
            title="Chapter One",
            blocks=[Block(type=BlockType.PARAGRAPH, text="It was dark.")],
        )
    ]


def test_render_podcast_returns_speaker_lines(tmp_path: Path):
    out = render_podcast("Test Book", _chapters(), PodcastLLM(), tmp_path)
    assert out.count("[[speaker:1]]") == 2
    assert out.count("[[speaker:2]]") == 2


def test_invalid_script_raises(tmp_path: Path):
    class BadLLM(PodcastLLM):
        def complete(self, user, *, system=None, max_tokens=2048):
            return "just prose with no tags"

    with pytest.raises(LLMError, match="speaker"):
        render_podcast("T", _chapters(), BadLLM(), tmp_path)


def test_untagged_lines_glue_to_the_previous_utterance(tmp_path: Path):
    class WrappedLLM(PodcastLLM):
        def complete(self, user, *, system=None, max_tokens=2048):
            return _SCRIPT.replace(
                "[[speaker:2]] Rainfall hit four hundred millimeters.",
                "[[speaker:2]] Rainfall hit\nfour hundred millimeters.",
            )

    out = render_podcast("T", _chapters(), WrappedLLM(), tmp_path)
    assert "[[speaker:2]] Rainfall hit four hundred millimeters." in out
    assert len(out.splitlines()) == 4  # the wrap did not become its own utterance


def test_oversized_source_is_truncated_with_a_warning(tmp_path: Path, caplog):
    huge = [
        Chapter(
            index=0,
            title="Chapter One",
            blocks=[Block(type=BlockType.PARAGRAPH, text="storm " * 60_000)],
        )
    ]
    llm = PodcastLLM()
    with caplog.at_level(logging.WARNING):
        render_podcast("Big Book", huge, llm, tmp_path)
    assert "truncat" in caplog.text.lower()  # no silent cap
    assert len(llm.prompts[0]) <= 300_000


def test_stage_writes_single_chapter(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(llm_pkg, "resolve_llm", lambda **kw: PodcastLLM())
    job = Job.create(tmp_path, JobConfig(source="x.epub", mode=Mode.PODCAST))
    doc = Document(meta=DocumentMeta(title="Test Book"), blocks=[])
    job.document_path.write_text(doc.model_dump_json(), encoding="utf-8")
    for ch in _chapters():
        (job.chapters_dir / f"{ch.index:03d}.json").write_text(
            ch.model_dump_json(), encoding="utf-8"
        )
    process_stage.run_stage(job)
    assert [p.name for p in sorted(job.processed_dir.glob("*.txt"))] == ["000.txt"]
    remaining = sorted(job.chapters_dir.glob("*.json"))
    assert len(remaining) == 1
    assert "Audio Digest" in remaining[0].read_text(encoding="utf-8")


def test_stage_keeps_old_chapters_when_the_script_is_invalid(tmp_path: Path, monkeypatch):
    """A failed run must not leave the job with neither old nor new chapters."""

    class BadLLM(PodcastLLM):
        def complete(self, user, *, system=None, max_tokens=2048):
            return "no tags here"

    monkeypatch.setattr(llm_pkg, "resolve_llm", lambda **kw: BadLLM())
    job = Job.create(tmp_path, JobConfig(source="x.epub", mode=Mode.PODCAST))
    doc = Document(meta=DocumentMeta(title="Test Book"), blocks=[])
    job.document_path.write_text(doc.model_dump_json(), encoding="utf-8")
    for ch in _chapters():
        (job.chapters_dir / f"{ch.index:03d}.json").write_text(
            ch.model_dump_json(), encoding="utf-8"
        )
    with pytest.raises(LLMError):
        process_stage.run_stage(job)
    kept = sorted(job.chapters_dir.glob("*.json"))
    assert len(kept) == 1
    assert "Chapter One" in kept[0].read_text(encoding="utf-8")


def _seed_job(tmp_path: Path, stale: int = 0) -> Job:
    job = Job.create(tmp_path, JobConfig(source="x.epub", mode=Mode.PODCAST))
    doc = Document(meta=DocumentMeta(title="Test Book"), blocks=[])
    job.document_path.write_text(doc.model_dump_json(), encoding="utf-8")
    for ch in _chapters():
        (job.chapters_dir / f"{ch.index:03d}.json").write_text(
            ch.model_dump_json(), encoding="utf-8"
        )
    job.processed_dir.mkdir(parents=True, exist_ok=True)
    for i in range(stale):
        (job.processed_dir / f"{i:03d}.txt").write_text(f"stale chapter {i}", encoding="utf-8")
    return job


def test_stage_replaces_stale_processed_text_from_a_previous_mode(tmp_path: Path, monkeypatch):
    """A --from-stage process re-run after switching modes must not leave orphan chapters."""
    monkeypatch.setattr(llm_pkg, "resolve_llm", lambda **kw: PodcastLLM())
    job = _seed_job(tmp_path, stale=3)
    process_stage.run_stage(job)
    assert [p.name for p in sorted(job.processed_dir.glob("*.txt"))] == ["000.txt"]
    assert "[[speaker:1]]" in (job.processed_dir / "000.txt").read_text(encoding="utf-8")


def test_stale_processed_text_survives_an_invalid_script(tmp_path: Path, monkeypatch):
    """The deletion carries the same ordering guarantee as the chapters/ replacement."""

    class BadLLM(PodcastLLM):
        def complete(self, user, *, system=None, max_tokens=2048):
            return "no tags here"

    monkeypatch.setattr(llm_pkg, "resolve_llm", lambda **kw: BadLLM())
    job = _seed_job(tmp_path, stale=3)
    with pytest.raises(LLMError):
        process_stage.run_stage(job)
    kept = sorted(p.name for p in job.processed_dir.glob("*.txt"))
    assert kept == ["000.txt", "001.txt", "002.txt"]


def test_markup_in_an_utterance_is_rejected(tmp_path: Path):
    class MarkdownLLM(PodcastLLM):
        def complete(self, user, *, system=None, max_tokens=2048):
            return _SCRIPT.replace(
                "[[speaker:2]] Storms are loud, and the data proves it.",
                "[[speaker:2]] Storms are **loud**, see <em>chart</em>.",
            )

    with pytest.raises(LLMError, match="markup"):
        render_podcast("T", _chapters(), MarkdownLLM(), tmp_path)


def test_script_with_only_one_speaker_is_rejected(tmp_path: Path):
    class MonologueLLM(PodcastLLM):
        def complete(self, user, *, system=None, max_tokens=2048):
            return "\n".join(f"[[speaker:1]] Point {i}." for i in range(5))

    with pytest.raises(LLMError, match="both hosts"):
        render_podcast("T", _chapters(), MonologueLLM(), tmp_path)


def test_utterances_get_the_rules_post_pass(tmp_path: Path):
    class SymbolLLM(PodcastLLM):
        def complete(self, user, *, system=None, max_tokens=2048):
            return _SCRIPT.replace(
                "[[speaker:2]] Rainfall hit four hundred millimeters.",
                "[[speaker:2]] Growth hit 40% and Fig. 3 shows it.",
            )

    out = render_podcast("T", _chapters(), SymbolLLM(), tmp_path)
    assert "40 percent" in out
    assert "Figure 3" in out
    assert out.splitlines()[-1].startswith("[[speaker:2]] ")  # the tag survives the pass


def test_rejected_script_is_evicted_so_the_next_run_regenerates(tmp_path: Path):
    """The trap: a deterministic key plus a cached bad script makes a job unrunnable forever."""

    class FlipFlopLLM(PodcastLLM):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def complete(self, user, *, system=None, max_tokens=2048):
            self.calls += 1
            if self.calls == 1:
                return _SCRIPT.replace("Storms are loud", "Storms are **loud**")
            return _SCRIPT

    llm = FlipFlopLLM()
    cache_dir = tmp_path / "cache"
    with pytest.raises(LLMError, match="markup"):
        render_podcast("T", _chapters(), llm, cache_dir)
    assert list(cache_dir.glob("*.txt")) == []  # the bad script did not survive

    out = render_podcast("T", _chapters(), llm, cache_dir)
    assert "[[speaker:2]] Storms are loud, and the data proves it." in out
    assert llm.calls == 2  # the second run really did re-generate


def test_accepted_script_stays_cached(tmp_path: Path):
    """Eviction must be scoped to rejection: a good script is still served from cache."""
    llm = PodcastLLM()
    cache_dir = tmp_path / "cache"
    render_podcast("T", _chapters(), llm, cache_dir)
    render_podcast("T", _chapters(), llm, cache_dir)
    assert len(llm.prompts) == 1
