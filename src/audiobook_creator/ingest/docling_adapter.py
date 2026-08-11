import logging
import os
import re
from pathlib import Path

from audiobook_creator.models import Block, BlockType, Document, DocumentMeta

logger = logging.getLogger(__name__)

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


def _plain_text(item) -> str:
    return " ".join((getattr(item, "text", "") or "").split())


_MD_SEPARATOR = re.compile(r"[|\s:\-]+")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def _markdown_table_to_prose(markdown: str) -> str:
    rows: list[str] = []
    for line in _HTML_COMMENT.sub("", markdown).splitlines():
        line = line.strip()
        if not line or _MD_SEPARATOR.fullmatch(line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
        if cells:
            rows.append(", ".join(cells))
    return ". ".join(rows) + ("." if rows else "")


def _table_text(item, dl_doc) -> str:
    # docling TableItem is a FloatingItem with no .text; its content is only
    # reachable via export_to_markdown(doc). Pictures must NOT take this path:
    # their export emits a diagnostic HTML comment, and captions arrive as
    # separate caption items already.
    text = _plain_text(item)
    if text:
        return text
    export = getattr(item, "export_to_markdown", None)
    if export is None:
        return ""
    try:
        markdown = export(dl_doc) or ""
    except Exception as exc:  # noqa: BLE001 - third-party export; losing one table beats aborting
        logger.warning("table markdown export failed; table content dropped: %s", exc)
        return ""
    return _markdown_table_to_prose(markdown)


def _save_picture(item, dl_doc, assets_dir: Path, index: int) -> str | None:
    """Save a picture item's bitmap; None when the item carries no usable image."""
    get_image = getattr(item, "get_image", None)
    if get_image is None:
        return None
    dest = assets_dir / f"fig-{index:03d}.png"
    try:
        image = get_image(dl_doc)
        if image is None:
            return None
        assets_dir.mkdir(parents=True, exist_ok=True)
        image.save(dest)
    except Exception as exc:  # noqa: BLE001 - third-party; a figure without art still narrates
        logger.debug("figure %d image extraction failed: %s", index, exc)
        return None
    return str(dest)


def document_from_docling(dl_doc, assets_dir: Path | None = None) -> Document:
    blocks: list[Block] = []
    title: str | None = None
    figures = 0
    image_failures = 0
    # iterate_items() yields (item, depth); depth is tree-traversal depth, not
    # heading rank — rank lives on item.level (verified against docling-core).
    for item, _depth in dl_doc.iterate_items():
        label = _label_value(item)
        if label not in _LABEL_MAP:
            continue
        block_type = _LABEL_MAP[label]
        if block_type is None:
            continue
        text = _table_text(item, dl_doc) if label == "table" else _plain_text(item)
        if label == "title" and title is None:
            title = text
        if not text and block_type is not BlockType.FIGURE:
            continue  # a picture with no caption is still content
        if block_type is BlockType.HEADING:
            level = getattr(item, "level", 1) or 1
            blocks.append(Block(type=block_type, text=text, level=int(level)))
        elif block_type is BlockType.FIGURE and assets_dir is not None:
            image_path = _save_picture(item, dl_doc, assets_dir, figures)
            figures += 1
            if image_path is None:
                image_failures += 1
            blocks.append(Block(type=block_type, text=text, image_path=image_path))
        else:
            blocks.append(Block(type=block_type, text=text))
    # Summarised rather than logged per figure: a picture-heavy PDF would otherwise emit
    # one warning per page for what is a single condition.
    if image_failures:
        logger.warning("%d of %d figure image(s) could not be extracted", image_failures, figures)
    meta = DocumentMeta(title=title or getattr(dl_doc, "name", None) or "Untitled")
    return Document(meta=meta, blocks=blocks)


def ingest_with_docling(source: str, assets_dir: Path) -> Document:
    # docling's layout model runs through torch.compile, and Inductor shells out to MSVC.
    # On a Windows box without cl.exe on PATH that aborts PDF ingest outright
    # (InvalidCxxCompiler: cl not found), so eager execution is the safe default here.
    # setdefault, not assignment: anyone who has set it deliberately keeps their value.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise RuntimeError(
            "PDF/DOCX/HTML ingestion requires Docling. Install it with: uv sync --extra pdf"
        ) from exc
    assets_dir.mkdir(parents=True, exist_ok=True)
    result = DocumentConverter().convert(source)
    return document_from_docling(result.document, assets_dir=assets_dir)
