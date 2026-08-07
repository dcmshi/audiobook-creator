import posixpath
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup

from audiobook_creator.models import Block, BlockType, Document, DocumentMeta

_CNT_NS = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
_OPF_NS = {"opf": "http://www.idpf.org/2007/opf", "dc": "http://purl.org/dc/elements/1.1/"}


def ingest_epub(path: Path, assets_dir: Path) -> Document:
    assets_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        opf_path = _opf_path(zf)
        opf_root = ET.fromstring(zf.read(opf_path).decode("utf-8"))
        opf_dir = posixpath.dirname(opf_path)

        meta = _metadata(opf_root)
        meta.cover_path = _extract_cover(zf, opf_root, opf_dir, assets_dir)

        blocks: list[Block] = []
        for href in _spine_hrefs(opf_root):
            xhtml = zf.read(posixpath.join(opf_dir, href) if opf_dir else href).decode("utf-8")
            blocks.extend(_blocks_from_xhtml(xhtml))
    return Document(meta=meta, blocks=blocks)


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


def _extract_cover(
    zf: zipfile.ZipFile, opf_root: ET.Element, opf_dir: str, assets_dir: Path
) -> str | None:
    for item in _manifest(opf_root).values():
        if "cover-image" in item.attrib.get("properties", ""):
            href = item.attrib["href"]
            src = posixpath.join(opf_dir, href) if opf_dir else href
            dest = assets_dir / f"cover{Path(href).suffix}"
            try:
                dest.write_bytes(zf.read(src))
            except KeyError:
                return None
            return str(dest)
    return None


_BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "table", "li"]


def _blocks_from_xhtml(xhtml: str) -> list[Block]:
    soup = BeautifulSoup(xhtml, "html.parser")
    body = soup.find("body")
    if body is None:
        return []
    blocks: list[Block] = []
    for el in body.find_all(_BLOCK_TAGS):
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
