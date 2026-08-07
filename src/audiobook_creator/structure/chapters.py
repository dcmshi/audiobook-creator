from audiobook_creator.models import Block, BlockType, Chapter, Document, Matter

_FRONT_KEYWORDS = {
    "contents", "table of contents", "title page", "copyright", "dedication",
    "preface", "foreword", "epigraph", "half title", "about this book",
}
_BACK_KEYWORDS = {
    "references", "bibliography", "index", "acknowledgments", "acknowledgements",
    "appendix", "notes", "glossary", "about the author", "endnotes",
}


def classify_matter(title: str) -> Matter:
    t = title.strip().lower()
    if any(t == k or t.startswith(k) for k in _BACK_KEYWORDS):
        return Matter.BACK
    if any(t == k or t.startswith(k) for k in _FRONT_KEYWORDS):
        return Matter.FRONT
    return Matter.BODY  # ambiguous -> body: extra audio beats missing content


def _split_at_level(doc: Document, max_level: int) -> list[Chapter]:
    chapters: list[Chapter] = []
    current_title = "Beginning"
    current_blocks: list[Block] = []

    def flush():
        if current_blocks or (chapters and current_title != "Beginning"):
            chapters.append(
                Chapter(
                    index=len(chapters),
                    title=current_title,
                    matter=classify_matter(current_title),
                    blocks=list(current_blocks),
                )
            )

    for block in doc.blocks:
        if block.type is BlockType.HEADING and (block.level or 1) <= max_level:
            flush()
            current_title = block.text
            current_blocks = []
        else:
            current_blocks.append(block)
    flush()
    return chapters


def split_chapters(doc: Document) -> list[Chapter]:
    chapters = _split_at_level(doc, max_level=1)
    if len(chapters) < 2:
        level2_count = sum(
            1 for b in doc.blocks if b.type is BlockType.HEADING and b.level == 2
        )
        if level2_count >= 2:
            chapters = _split_at_level(doc, max_level=2)
    return chapters
