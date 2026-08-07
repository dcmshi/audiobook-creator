from pathlib import Path

from audiobook_creator.ingest.epub import _blocks_from_xhtml, ingest_epub
from audiobook_creator.models import BlockType

_LIST_XHTML = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Lists</title></head>
<body>
<h1>Findings</h1>
<ul><li>First finding.</li><li>Second finding.</li></ul>
<ol><li>Step one.</li></ol>
</body></html>
"""

_NESTED_XHTML = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Nested</title></head>
<body>
<table><tr><td><p>Cell prose.</p></td><td>2026</td></tr></table>
<ul><li><p>Wrapped item.</p></li></ul>
</body></html>
"""


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


def test_list_items_become_paragraphs():
    blocks = _blocks_from_xhtml(_LIST_XHTML)
    paragraphs = [b.text for b in blocks if b.type is BlockType.PARAGRAPH]
    assert paragraphs == ["First finding.", "Second finding.", "Step one."]


def test_table_cell_prose_appears_exactly_once():
    blocks = _blocks_from_xhtml(_NESTED_XHTML)
    assert sum("Cell prose." in b.text for b in blocks) == 1
    tables = [b for b in blocks if b.type is BlockType.TABLE]
    assert len(tables) == 1
    assert "Cell prose." in tables[0].text


def test_list_item_wrapping_paragraph_not_duplicated():
    blocks = _blocks_from_xhtml(_NESTED_XHTML)
    wrapped = [b for b in blocks if b.text == "Wrapped item."]
    assert len(wrapped) == 1
    assert wrapped[0].type is BlockType.PARAGRAPH
