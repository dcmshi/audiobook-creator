from pathlib import Path

from audiobook_creator.core.job import Job
from audiobook_creator.models import Block, BlockType, Chapter, JobConfig
from audiobook_creator.process import stage as process_stage
from audiobook_creator.process.llm.base import LLMError
from audiobook_creator.process.llm_verbatim import make_llm_normalizer


class GoodLLM:
    name = "fake"
    model = "m"

    def complete(self, user, *, system=None, max_tokens=2048):
        return user.replace("Dr.", "Doctor")

    def describe_image(self, p, q, *, max_tokens=1024):
        return ""


class SummarizingLLM(GoodLLM):
    def complete(self, user, *, system=None, max_tokens=2048):
        return "short."


class BrokenLLM(GoodLLM):
    def complete(self, user, *, system=None, max_tokens=2048):
        raise LLMError("down")


class ExpandingLLM(GoodLLM):
    """Correct normalization of DENSE_SOURCE, as produced by a real model: ratio ~1.79."""

    def complete(self, user, *, system=None, max_tokens=2048):
        return (
            "Smith and colleagues found that twelve percent of the one thousand "
            "four hundred fifty samples exceeded five milligrams per liter."
        )


class PaddingLLM(GoodLLM):
    def complete(self, user, *, system=None, max_tokens=2048):
        return user + " " + "Here is the normalized text you asked for. " * 6


DENSE_SOURCE = "Smith et al. (2020) found that 12% of the 1,450 samples exceeded 5 mg/L."


def test_normalizer_applies_llm_output(tmp_path: Path):
    fn = make_llm_normalizer(GoodLLM(), tmp_path)
    out = fn("Dr. Smith arrived at the scene today.")
    assert out == "Doctor Smith arrived at the scene today."


def test_summarizing_output_falls_back_to_rules(tmp_path: Path):
    fn = make_llm_normalizer(SummarizingLLM(), tmp_path)
    text = "Results were strong [12]. The rain fell for a very long time across the plains."
    out = fn(text)
    assert "short." not in out
    assert "[12]" not in out  # rule fallback still normalized


def test_dense_expansion_is_accepted(tmp_path: Path):
    """Spelling out numbers and units legitimately inflates length; that is the job."""
    fn = make_llm_normalizer(ExpandingLLM(), tmp_path)
    out = fn(DENSE_SOURCE)
    assert out.startswith("Smith and colleagues found that twelve percent")
    assert len(out) / len(DENSE_SOURCE) > 1.5  # the old ceiling would have discarded this


def test_padding_still_falls_back_to_rules(tmp_path: Path):
    fn = make_llm_normalizer(PaddingLLM(), tmp_path)
    out = fn(DENSE_SOURCE)
    assert "Here is the normalized text" not in out
    assert "and colleagues" in out  # rule fallback ran


def test_llm_error_falls_back_to_rules(tmp_path: Path):
    fn = make_llm_normalizer(BrokenLLM(), tmp_path)
    assert "percent" in fn("Growth of 40% overall this year across all divisions.")


def test_stage_records_llm_backend(tmp_path: Path, monkeypatch):
    from audiobook_creator.process import llm as llm_pkg

    monkeypatch.setattr(
        llm_pkg,
        "resolve_llm",
        lambda *, local_only, use_llm, provider=None: GoodLLM() if use_llm else None,
    )
    job = Job.create(tmp_path, JobConfig(source="x.epub"))
    ch = Chapter(
        index=0,
        title="One",
        blocks=[Block(type=BlockType.PARAGRAPH, text="Dr. Smith arrived here today, ready.")],
    )
    (job.chapters_dir / "000.json").write_text(ch.model_dump_json(), encoding="utf-8")
    process_stage.run_stage(job)
    assert "llm:fake" in job.state.backends_used
    assert "Doctor Smith" in (job.processed_dir / "000.txt").read_text(encoding="utf-8")
