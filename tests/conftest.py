import zipfile
from pathlib import Path

import pytest

_CONTAINER = """<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""

_OPF = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">test-book-001</dc:identifier>
    <dc:title>Test Book</dc:title>
    <dc:creator>Jane Doe</dc:creator>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
    <item id="c3" href="refs.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="c1"/>
    <itemref idref="c2"/>
    <itemref idref="c3"/>
  </spine>
</package>
"""

_NAV = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>Nav</title></head>
<body><nav epub:type="toc"><ol>
<li><a href="ch1.xhtml">Chapter One</a></li>
<li><a href="ch2.xhtml">Chapter Two</a></li>
<li><a href="refs.xhtml">References</a></li>
</ol></nav></body></html>
"""

_CH1 = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Ch 1</title></head>
<body>
<h1>Chapter One</h1>
<p>It was a dark and stormy night.</p>
<p>The rain fell in torrents, at 40% intensity.</p>
</body></html>
"""

_CH2 = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Ch 2</title></head>
<body>
<h1>Chapter Two</h1>
<p>Morning came quietly.</p>
<table><tr><td>Year</td><td>2026</td></tr></table>
</body></html>
"""

_REFS = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Refs</title></head>
<body>
<h1>References</h1>
<p>Doe, J. (2026). A study of storms.</p>
</body></html>
"""


@pytest.fixture(autouse=True)
def no_implicit_llm(monkeypatch):
    """Keep the suite hermetic and deterministic.

    The process stage resolves an LLM on its own now, and this project's documented dev setup
    runs Ollama locally — so without this the stage tests make real inference calls, taking
    ~50s and making assertions depend on model output. Tests that exercise the LLM path
    monkeypatch resolve_llm (or delenv this) explicitly.
    """
    monkeypatch.setenv("ABC_LLM", "none")


# ch1 with a figure in it: once as a bare sibling of the paragraphs, once wrapped in the
# text-free <p class="image"> that EPUB conversion tools commonly emit.
_CH1_WITH_IMG = _CH1.replace(
    "<p>It was a dark and stormy night.</p>",
    '<p>It was a dark and stormy night.</p>\n<img src="pic.png" alt="A storm chart"/>',
)

_CH1_WITH_WRAPPED_IMG = _CH1.replace(
    "<p>It was a dark and stormy night.</p>",
    '<p>It was a dark and stormy night.</p>\n'
    '<p class="image"><img src="pic.png" alt="A storm chart"/></p>',
)

# EPUB2 names the cover indirectly: a <meta name="cover"> pointing at a manifest id, with no
# EPUB3 properties="cover-image" anywhere.
_OPF_EPUB2_COVER = _OPF.replace(
    "<dc:language>en</dc:language>",
    '<dc:language>en</dc:language>\n    <meta name="cover" content="cover-img"/>',
).replace(
    '<item id="nav"',
    '<item id="cover-img" href="cover.jpg" media-type="image/jpeg"/>\n    <item id="nav"',
)


def _build_epub(
    epub_path: Path,
    *,
    ch1: str = _CH1,
    opf: str = _OPF,
    extra: dict[str, bytes] | None = None,
) -> Path:
    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", _CONTAINER)
        zf.writestr("OEBPS/content.opf", opf)
        zf.writestr("OEBPS/nav.xhtml", _NAV)
        zf.writestr("OEBPS/ch1.xhtml", ch1)
        zf.writestr("OEBPS/ch2.xhtml", _CH2)
        zf.writestr("OEBPS/refs.xhtml", _REFS)
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return epub_path


_OPF_EPUB3_COVER = _OPF.replace(
    '<item id="nav"',
    '<item id="cover-img" href="cover.jpg" media-type="image/jpeg"'
    ' properties="cover-image"/>\n    <item id="nav"',
)


def _epub_with_extra_files(
    tmp_path: Path,
    *,
    images: dict[str, bytes] | None = None,
    epub2_cover: bool = False,
    epub3_cover: bool = False,
    wrap_img: bool = False,
) -> Path:
    """The make_epub book plus optional image entries, an <img> in ch1, or either cover style."""
    extra = dict(images or {})
    opf = _OPF
    if epub2_cover or epub3_cover:
        opf = _OPF_EPUB2_COVER if epub2_cover else _OPF_EPUB3_COVER
        extra.setdefault("OEBPS/cover.jpg", b"\xff\xd8\xffJPEGBYTES")
    ch1 = _CH1
    if images:
        ch1 = _CH1_WITH_WRAPPED_IMG if wrap_img else _CH1_WITH_IMG
    return _build_epub(
        tmp_path / "extra-book.epub",
        ch1=ch1,
        opf=opf,
        extra=extra,
    )


@pytest.fixture
def make_epub(tmp_path: Path):
    def _make() -> Path:
        return _build_epub(tmp_path / "test-book.epub")

    return _make
