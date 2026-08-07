from pathlib import Path

from audiobook_creator.ingest.epub import ingest_epub
from audiobook_creator.models import BlockType


def test_metadata_extracted(make_epub, tmp_path: Path):
    doc = ingest_epub(make_epub(), tmp_path / "assets")
    assert doc.meta.title == "Test Book"
    assert doc.meta.author == "Jane Doe"


def test_blocks_in_spine_order(make_epub, tmp_path: Path):
    doc = ingest_epub(make_epub(), tmp_path / "assets")
    headings = [b.text for b in doc.blocks if b.type is BlockType.HEADING]
    assert headings == ["Chapter One", "Chapter Two", "References"]
    assert all(b.level == 1 for b in doc.blocks if b.type is BlockType.HEADING)

    first_heading = next(i for i, b in enumerate(doc.blocks) if b.type is BlockType.HEADING)
    paragraphs_after = [
        b.text for b in doc.blocks[first_heading:] if b.type is BlockType.PARAGRAPH
    ]
    assert paragraphs_after[0] == "It was a dark and stormy night."


def test_table_becomes_table_block(make_epub, tmp_path: Path):
    doc = ingest_epub(make_epub(), tmp_path / "assets")
    tables = [b for b in doc.blocks if b.type is BlockType.TABLE]
    assert len(tables) == 1
    assert "2026" in tables[0].text
