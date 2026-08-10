from collections.abc import Callable

from audiobook_creator.core.job import Job
from audiobook_creator.models import Chapter, Matter, Mode
from audiobook_creator.process import llm as llm_pkg
from audiobook_creator.process.llm.base import LLMClient
from audiobook_creator.process.llm_verbatim import make_llm_normalizer
from audiobook_creator.process.rewrite import render_rewrite
from audiobook_creator.process.verbatim import has_speakable_blocks, render_chapter_text


def _record_backend(job: Job, client: LLMClient) -> None:
    used = f"llm:{client.name}"
    if used not in job.state.backends_used:
        job.state.backends_used.append(used)
        job.save()


def _make_renderer(job: Job) -> Callable[[Chapter], str]:
    cfg = job.state.config
    # A PrivacyError from resolve_llm is deliberate: a local_only job that asked for a network
    # provider must fail loudly rather than quietly narrate through the rule path.
    if cfg.mode is Mode.VERBATIM:
        client = llm_pkg.resolve_llm(
            local_only=cfg.local_only, use_llm=cfg.use_llm, provider=cfg.llm_provider
        )
        if client is None:
            return render_chapter_text
        _record_backend(job, client)
        normalizer = make_llm_normalizer(client, job.dir / "llm-cache")
        return lambda chapter: render_chapter_text(chapter, normalizer=normalizer)
    if cfg.mode is Mode.REWRITE:
        # Rewrite has no rule-based equivalent, so use_llm=True regardless of config: there is
        # nothing to degrade to. Preflight should have caught this, but the stage still refuses
        # rather than producing a book of raw table cells.
        client = llm_pkg.resolve_llm(
            local_only=cfg.local_only, use_llm=True, provider=cfg.llm_provider
        )
        if client is None:
            raise RuntimeError(
                "rewrite mode needs an LLM (start Ollama, or pass --llm anthropic / --llm kimi)"
            )
        _record_backend(job, client)
        cache_dir = job.dir / "llm-cache"
        return lambda chapter: render_rewrite(chapter, client, cache_dir)
    raise NotImplementedError(
        f"mode {cfg.mode.value!r} lands later in Plan 2; 'verbatim' and 'rewrite' work today"
    )


def run_stage(job: Job) -> None:
    render = _make_renderer(job)
    job.processed_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    # Every titled chapter renders a title line, so counting files cannot tell us
    # whether the book has any prose at all — track block-derived text separately.
    any_prose = False
    for path in sorted(job.chapters_dir.glob("*.json")):
        chapter = Chapter.model_validate_json(path.read_text(encoding="utf-8"))
        if chapter.matter is not Matter.BODY:
            continue
        text = render(chapter)
        if not text.strip():
            continue
        any_prose = any_prose or has_speakable_blocks(chapter)
        (job.processed_dir / f"{chapter.index:03d}.txt").write_text(text, encoding="utf-8")
        written += 1
    if written == 0 or not any_prose:
        raise ValueError("no body chapters produced speakable text")
