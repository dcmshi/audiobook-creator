from collections.abc import Callable

from audiobook_creator.core.job import Job
from audiobook_creator.models import Chapter, Document, Matter, Mode
from audiobook_creator.process import llm as llm_pkg
from audiobook_creator.process.llm.base import LLMClient
from audiobook_creator.process.llm_verbatim import make_llm_normalizer
from audiobook_creator.process.podcast import render_podcast
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
        client = _require_client(job, "rewrite")
        cache_dir = job.dir / "llm-cache"
        return lambda chapter: render_rewrite(chapter, client, cache_dir)
    raise NotImplementedError(f"mode {cfg.mode.value!r} has no renderer")


def _require_client(job: Job, mode_name: str) -> LLMClient:
    cfg = job.state.config
    client = llm_pkg.resolve_llm(
        local_only=cfg.local_only, use_llm=True, provider=cfg.llm_provider
    )
    if client is None:
        raise RuntimeError(
            f"{mode_name} mode needs an LLM (start Ollama, or pass --llm anthropic / --llm kimi)"
        )
    _record_backend(job, client)
    return client


def _body_chapters(job: Job) -> list[Chapter]:
    chapters = []
    for path in sorted(job.chapters_dir.glob("*.json")):
        chapter = Chapter.model_validate_json(path.read_text(encoding="utf-8"))
        if chapter.matter is Matter.BODY:
            chapters.append(chapter)
    return chapters


def _run_podcast(job: Job) -> None:
    client = _require_client(job, "podcast")
    title = "Untitled"
    if job.document_path.is_file():
        title = Document.model_validate_json(
            job.document_path.read_text(encoding="utf-8")
        ).meta.title
    chapters = _body_chapters(job)
    if not chapters:
        raise ValueError("no body chapters to build a podcast from")
    # An invalid script raises here, before anything on disk is touched.
    script = render_podcast(title, chapters, client, job.dir / "llm-cache")
    job.processed_dir.mkdir(parents=True, exist_ok=True)
    (job.processed_dir / "000.txt").write_text(script, encoding="utf-8")
    # Only now replace the chapter set: a failed run must leave the job with its old chapters
    # rather than neither set. These are re-derivable with --from-stage structure.
    digest = Chapter(index=0, title=f"{title} — Audio Digest", blocks=[])
    for path in job.chapters_dir.glob("*.json"):
        path.unlink()
    (job.chapters_dir / "000.json").write_text(digest.model_dump_json(), encoding="utf-8")


def run_stage(job: Job) -> None:
    if job.state.config.mode is Mode.PODCAST:
        _run_podcast(job)
        return
    render = _make_renderer(job)
    # Rewrite verbalizes tables and figures, so they count towards "this book has narration".
    include_visual = job.state.config.mode is Mode.REWRITE
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
        any_prose = any_prose or has_speakable_blocks(chapter, include_visual=include_visual)
        (job.processed_dir / f"{chapter.index:03d}.txt").write_text(text, encoding="utf-8")
        written += 1
    if written == 0 or not any_prose:
        raise ValueError("no body chapters produced speakable text")
