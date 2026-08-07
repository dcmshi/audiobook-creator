from audiobook_creator.models import Block, BlockType, Chapter, Document, Matter

# Ordinary English words must match the whole title. "Notes on a Scandal" and
# "Index Funds Explained" are body chapters, and misfiling one here deletes it:
# the process stage narrates body matter only.
_FRONT_EXACT = {
    "contents", "copyright", "dedication", "preface", "foreword", "epigraph",
}
_BACK_EXACT = {
    "references", "index", "notes", "glossary", "endnotes",
}

# Publishing jargon, implausible as the opening words of a body chapter title, so
# these may match a prefix and still catch "Appendix A: Data".
_FRONT_PREFIX = {
    "table of contents", "title page", "half title", "about this book",
}
_BACK_PREFIX = {
    "bibliography", "acknowledgments", "acknowledgements", "appendix", "about the author",
}


def classify_matter(title: str) -> Matter:
    t = title.strip().lower()
    if t in _BACK_EXACT or any(t.startswith(k) for k in _BACK_PREFIX):
        return Matter.BACK
    if t in _FRONT_EXACT or any(t.startswith(k) for k in _FRONT_PREFIX):
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
