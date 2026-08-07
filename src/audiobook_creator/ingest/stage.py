from pathlib import Path

from audiobook_creator.core.job import Job
from audiobook_creator.ingest.docling_adapter import ingest_with_docling
from audiobook_creator.ingest.epub import ingest_epub
from audiobook_creator.models import Document


def ingest(source: str, assets_dir: Path) -> Document:
    if not source.startswith(("http://", "https://")) and source.lower().endswith(".epub"):
        return ingest_epub(Path(source), assets_dir)
    return ingest_with_docling(source, assets_dir)


def run_stage(job: Job) -> None:
    doc = ingest(job.state.config.source, job.assets_dir)
    if not doc.blocks:
        raise ValueError(f"ingestion produced no content from {job.state.config.source!r}")
    job.document_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
