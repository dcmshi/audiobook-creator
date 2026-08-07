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


@pytest.fixture
def make_epub(tmp_path: Path):
    def _make() -> Path:
        epub_path = tmp_path / "test-book.epub"
        with zipfile.ZipFile(epub_path, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
            zf.writestr("META-INF/container.xml", _CONTAINER)
            zf.writestr("OEBPS/content.opf", _OPF)
            zf.writestr("OEBPS/nav.xhtml", _NAV)
            zf.writestr("OEBPS/ch1.xhtml", _CH1)
            zf.writestr("OEBPS/ch2.xhtml", _CH2)
            zf.writestr("OEBPS/refs.xhtml", _REFS)
        return epub_path

    return _make
