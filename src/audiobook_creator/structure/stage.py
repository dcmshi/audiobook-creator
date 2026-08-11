import logging

from audiobook_creator.core.job import Job
from audiobook_creator.models import Document
from audiobook_creator.process import llm as llm_pkg
from audiobook_creator.structure.chapters import refine_matter_with_llm, split_chapters

logger = logging.getLogger(__name__)


def run_stage(job: Job) -> None:
    cfg = job.state.config
    doc = Document.model_validate_json(job.document_path.read_text(encoding="utf-8"))
    chapters = split_chapters(doc)
    if not chapters:
        raise ValueError("structure stage produced no chapters")
    # A PrivacyError propagates: local_only asking for a network provider is a job-level
    # mistake, not something to paper over with the rule-based classification.
    client = llm_pkg.resolve_llm(
        local_only=cfg.local_only, use_llm=cfg.use_llm, provider=cfg.llm_provider
    )
    if client is not None:
        chapters = refine_matter_with_llm(chapters, client)
    job.chapters_dir.mkdir(parents=True, exist_ok=True)
    for chapter in chapters:
        path = job.chapters_dir / f"{chapter.index:03d}.json"
        path.write_text(chapter.model_dump_json(indent=2), encoding="utf-8")
