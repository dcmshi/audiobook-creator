from audiobook_creator.core.job import Job
from audiobook_creator.models import Document
from audiobook_creator.structure.chapters import split_chapters


def run_stage(job: Job) -> None:
    doc = Document.model_validate_json(job.document_path.read_text(encoding="utf-8"))
    chapters = split_chapters(doc)
    if not chapters:
        raise ValueError("structure stage produced no chapters")
    job.chapters_dir.mkdir(parents=True, exist_ok=True)
    for chapter in chapters:
        path = job.chapters_dir / f"{chapter.index:03d}.json"
        path.write_text(chapter.model_dump_json(indent=2), encoding="utf-8")
