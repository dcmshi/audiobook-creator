from pathlib import Path
from types import SimpleNamespace

import pytest

from audiobook_creator.core.job import Job
from audiobook_creator.ingest.docling_adapter import document_from_docling
from audiobook_creator.ingest.stage import ingest, run_stage
from audiobook_creator.models import BlockType, Document, JobConfig


def _item(label: str, text: str = "", level: int | None = None):
    ns = SimpleNamespace(label=SimpleNamespace(value=label), text=text)
    if level is not None:
        ns.level = level
    return ns


class FakeDoclingDoc:
    name = "My Paper"

    def iterate_items(self):
        items = [
            _item("title", "My Paper"),
            _item("page_header", "Journal of Storms 2026"),
            _item("section_header", "Introduction", level=1),
            _item("text", "Storms are loud."),
            _item("footnote", "1. See appendix."),
            _item("table", "Year 2026 Rain 400mm"),
            _item("picture"),
            _item("caption", "Figure 1: A storm."),
            _item("page_footer", "Page 3"),
        ]
        return [(i, 0) for i in items]


def test_labels_mapped_and_furniture_dropped():
    doc = document_from_docling(FakeDoclingDoc())
    types = [b.type for b in doc.blocks]
    assert BlockType.HEADING in types
    assert BlockType.TABLE in types
    assert BlockType.FIGURE in types
    assert BlockType.CAPTION in types
    assert BlockType.FOOTNOTE in types
    texts = " ".join(b.text for b in doc.blocks)
    assert "Journal of Storms" not in texts  # page_header dropped
    assert "Page 3" not in texts  # page_footer dropped
    assert doc.meta.title == "My Paper"


def test_heading_levels_preserved():
    doc = document_from_docling(FakeDoclingDoc())
    intro = next(b for b in doc.blocks if b.text == "Introduction")
    assert intro.type is BlockType.HEADING
    assert intro.level == 1


def test_dispatcher_routes_epub(make_epub, tmp_path: Path):
    doc = ingest(str(make_epub()), tmp_path / "assets")
    assert isinstance(doc, Document)
    assert doc.meta.title == "Test Book"


def test_dispatcher_pdf_without_docling_gives_actionable_error(tmp_path: Path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_docling(name, *args, **kwargs):
        if name.startswith("docling"):
            raise ImportError("No module named 'docling'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_docling)
    with pytest.raises(RuntimeError, match="--extra pdf"):
        ingest("paper.pdf", tmp_path / "assets")


def test_run_stage_writes_document_json(make_epub, tmp_path: Path):
    job = Job.create(tmp_path / "jobs", JobConfig(source=str(make_epub())))
    run_stage(job)
    assert job.document_path.is_file()
    doc = Document.model_validate_json(job.document_path.read_text(encoding="utf-8"))
    assert doc.meta.title == "Test Book"
