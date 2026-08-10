import itertools
import logging
import posixpath
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from pathlib import Path

from bs4 import BeautifulSoup

from audiobook_creator.models import Block, BlockType, Document, DocumentMeta

logger = logging.getLogger(__name__)

_CNT_NS = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
_OPF_NS = {"opf": "http://www.idpf.org/2007/opf", "dc": "http://purl.org/dc/elements/1.1/"}

# A missing, corrupt, or truncated member, or an unwritable destination. Neither BadZipFile
# nor EOFError is an OSError, so both need naming. An unreadable image costs its picture,
# never the book.
_ASSET_FAILURES = (KeyError, OSError, zipfile.BadZipFile, EOFError)


def ingest_epub(path: Path, assets_dir: Path) -> Document:
    assets_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        opf_path = _opf_path(zf)
        opf_root = ET.fromstring(zf.read(opf_path).decode("utf-8"))
        opf_dir = posixpath.dirname(opf_path)

        failures: list[str] = []
        meta = _metadata(opf_root)
        meta.cover_path = _extract_cover(zf, opf_root, opf_dir, assets_dir, failures)

        blocks: list[Block] = []
        figures = itertools.count()
        for href in _spine_hrefs(opf_root):
            full_href = posixpath.join(opf_dir, href) if opf_dir else href
            xhtml = zf.read(full_href).decode("utf-8")

            # An <img src> resolves against its own document's directory, which is not
            # always the OPF's — bind it per spine item rather than reusing opf_dir.
            def save_image(src: str, _base: str = posixpath.dirname(full_href)) -> str | None:
                return _save_asset(zf, _base, src, assets_dir, next(figures), failures)

            blocks.extend(_blocks_from_xhtml(xhtml, save_image))
    # One line per book rather than per asset: a corrupt archive tends to fail in bulk, and
    # a silently image-less figure is exactly what leaves vision with nothing to describe.
    if failures:
        shown = ", ".join(failures[:5])
        logger.warning(
            "%d EPUB asset(s) could not be extracted; affected figures have no image (%s%s)",
            len(failures),
            shown,
            ", ..." if len(failures) > 5 else "",
        )
    return Document(meta=meta, blocks=blocks)


def _save_asset(
    zf: zipfile.ZipFile,
    base_dir: str,
    src: str,
    assets_dir: Path,
    index: int,
    failures: list[str],
) -> str | None:
    """Copy one referenced image out of the zip; None when it is not a packaged file."""
    if not src or "://" in src or src.startswith("data:"):
        return None
    target = posixpath.normpath(posixpath.join(base_dir, src) if base_dir else src)
    dest = assets_dir / f"fig-{index:03d}{Path(target).suffix or '.img'}"
    try:
        dest.write_bytes(zf.read(target))
    except _ASSET_FAILURES as exc:
        logger.debug("EPUB asset %r could not be extracted: %s", target, exc)
        failures.append(target)
        return None
    return str(dest)


def _opf_path(zf: zipfile.ZipFile) -> str:
    container = ET.fromstring(zf.read("META-INF/container.xml").decode("utf-8"))
    rootfile = container.find(".//c:rootfile", _CNT_NS)
    if rootfile is None:
        raise ValueError("EPUB has no rootfile in META-INF/container.xml")
    return rootfile.attrib["full-path"]


def _metadata(opf_root: ET.Element) -> DocumentMeta:
    title_el = opf_root.find(".//dc:title", _OPF_NS)
    author_el = opf_root.find(".//dc:creator", _OPF_NS)
    return DocumentMeta(
        title=(title_el.text or "Untitled").strip() if title_el is not None else "Untitled",
        author=author_el.text.strip() if author_el is not None and author_el.text else None,
    )


def _manifest(opf_root: ET.Element) -> dict[str, ET.Element]:
    return {
        item.attrib["id"]: item
        for item in opf_root.findall(".//opf:manifest/opf:item", _OPF_NS)
    }


def _spine_hrefs(opf_root: ET.Element) -> list[str]:
    manifest = _manifest(opf_root)
    hrefs: list[str] = []
    for itemref in opf_root.findall(".//opf:spine/opf:itemref", _OPF_NS):
        item = manifest.get(itemref.attrib["idref"])
        if item is not None and "nav" not in item.attrib.get("properties", ""):
            hrefs.append(item.attrib["href"])
    return hrefs


def _cover_item(opf_root: ET.Element) -> ET.Element | None:
    manifest = _manifest(opf_root)
    for item in manifest.values():
        if "cover-image" in item.attrib.get("properties", ""):
            return item
    # EPUB2 has no cover-image property: metadata names the manifest id instead.
    for meta in opf_root.findall(".//opf:meta", _OPF_NS):
        if meta.attrib.get("name") == "cover":
            return manifest.get(meta.attrib.get("content", ""))
    return None


def _extract_cover(
    zf: zipfile.ZipFile,
    opf_root: ET.Element,
    opf_dir: str,
    assets_dir: Path,
    failures: list[str],
) -> str | None:
    item = _cover_item(opf_root)
    if item is None:
        return None
    href = item.attrib["href"]
    src = posixpath.join(opf_dir, href) if opf_dir else href
    dest = assets_dir / f"cover{Path(href).suffix}"
    try:
        dest.write_bytes(zf.read(src))
    except _ASSET_FAILURES as exc:
        logger.debug("EPUB cover %r could not be extracted: %s", src, exc)
        failures.append(src)
        return None
    return str(dest)


_BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "table", "li"]
_CAPTURED_TAGS = [*_BLOCK_TAGS, "img"]


def _figure_block(img, save_image: Callable[[str], str | None] | None) -> Block | None:
    """A FIGURE for this <img>, or None when its text is already narrated by an ancestor.

    `<p class="image"><img/></p>` is a dominant real-world wrapper, and such a wrapper
    flattens to empty text — it is skipped as a block, so the figure would vanish with it.
    A wrapper carrying real prose keeps the anti-duplication guard instead: that prose is
    narrated from the block, and the image stays folded into it.
    """
    wrapper = img.find_parent(_BLOCK_TAGS)
    if wrapper is not None and wrapper.get_text(separator=" ").strip():
        return None
    # A void element, so there is nothing to flatten: the alt text is the caption.
    return Block(
        type=BlockType.FIGURE,
        text=" ".join(img.get("alt", "").split()),
        image_path=save_image(img.get("src", "")) if save_image else None,
    )


def _blocks_from_xhtml(
    xhtml: str, save_image: Callable[[str], str | None] | None = None
) -> list[Block]:
    soup = BeautifulSoup(xhtml, "html.parser")
    body = soup.find("body")
    if body is None:
        return []
    blocks: list[Block] = []
    for el in body.find_all(_CAPTURED_TAGS):
        if el.name == "img":
            figure = _figure_block(el, save_image)
            if figure is not None:
                blocks.append(figure)
            continue
        # find_all recurses, so a <p> inside a <td> or an <li> is already carried by
        # its ancestor's flattened text; emitting it again narrates the prose twice.
        if el.find_parent(_BLOCK_TAGS) is not None:
            continue
        text = " ".join(el.get_text(separator=" ").split())
        if not text:
            continue
        if el.name in ("p", "li"):
            blocks.append(Block(type=BlockType.PARAGRAPH, text=text))
        elif el.name == "table":
            blocks.append(Block(type=BlockType.TABLE, text=text))
        else:
            blocks.append(Block(type=BlockType.HEADING, text=text, level=int(el.name[1])))
    return blocks
