from pathlib import Path

from audiobook_creator.models import Block, BlockType, Document, DocumentMeta

# docling label value -> our BlockType; None = drop (page furniture)
_LABEL_MAP: dict[str, BlockType | None] = {
    "title": BlockType.HEADING,
    "section_header": BlockType.HEADING,
    "text": BlockType.PARAGRAPH,
    "paragraph": BlockType.PARAGRAPH,
    "list_item": BlockType.PARAGRAPH,
    "table": BlockType.TABLE,
    "picture": BlockType.FIGURE,
    "footnote": BlockType.FOOTNOTE,
    "caption": BlockType.CAPTION,
    "page_header": None,
    "page_footer": None,
    "page_number": None,
}


def _label_value(item) -> str:
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label))


def _item_text(item, dl_doc) -> str:
    # TableItem/PictureItem are FloatingItems and carry no .text; a table's
    # content is only reachable via export_to_markdown(doc).
    text = getattr(item, "text", "") or ""
    if not text:
        export = getattr(item, "export_to_markdown", None)
        if export is not None:
            try:
                text = export(dl_doc) or ""
            except Exception:
                text = ""
    return " ".join(text.split())


def document_from_docling(dl_doc) -> Document:
    blocks: list[Block] = []
    title: str | None = None
    # iterate_items() yields (item, depth); depth is tree-traversal depth, not
    # heading rank — rank lives on item.level (verified against docling-core).
    for item, _depth in dl_doc.iterate_items():
        label = _label_value(item)
        if label not in _LABEL_MAP:
            continue
        block_type = _LABEL_MAP[label]
        if block_type is None:
            continue
        text = _item_text(item, dl_doc)
        if label == "title" and title is None:
            title = text
        if not text and block_type is not BlockType.FIGURE:
            continue  # a picture with no caption is still content
        if block_type is BlockType.HEADING:
            level = getattr(item, "level", 1) or 1
            blocks.append(Block(type=block_type, text=text, level=int(level)))
        else:
            blocks.append(Block(type=block_type, text=text))
    meta = DocumentMeta(title=title or getattr(dl_doc, "name", None) or "Untitled")
    return Document(meta=meta, blocks=blocks)


def ingest_with_docling(source: str, assets_dir: Path) -> Document:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise RuntimeError(
            "PDF/DOCX/HTML ingestion requires Docling. Install it with: uv sync --extra pdf"
        ) from exc
    result = DocumentConverter().convert(source)
    return document_from_docling(result.document)
