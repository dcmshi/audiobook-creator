from audiobook_creator.core.job import Job
from audiobook_creator.models import Chapter, Matter, Mode
from audiobook_creator.process.verbatim import has_speakable_blocks, render_chapter_text


def run_stage(job: Job) -> None:
    mode = job.state.config.mode
    if mode is not Mode.VERBATIM:
        raise NotImplementedError(
            f"mode {mode.value!r} lands in Plan 2 (LLM layer); only 'verbatim' works today"
        )
    job.processed_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    # Every titled chapter renders a title line, so counting files cannot tell us
    # whether the book has any prose at all — track block-derived text separately.
    any_prose = False
    for path in sorted(job.chapters_dir.glob("*.json")):
        chapter = Chapter.model_validate_json(path.read_text(encoding="utf-8"))
        if chapter.matter is not Matter.BODY:
            continue
        text = render_chapter_text(chapter)
        if not text.strip():
            continue
        any_prose = any_prose or has_speakable_blocks(chapter)
        (job.processed_dir / f"{chapter.index:03d}.txt").write_text(text, encoding="utf-8")
        written += 1
    if written == 0 or not any_prose:
        raise ValueError("no body chapters produced speakable text")
