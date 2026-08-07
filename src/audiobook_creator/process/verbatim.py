from audiobook_creator.models import BlockType, Chapter
from audiobook_creator.process.rules import normalize

PAUSE = "[[pause]]"

# Verbatim v1 contract: read prose and captions; skip tables, figures, and
# footnote bodies (their inline markers are stripped by normalize()).
_SKIPPED = {BlockType.TABLE, BlockType.FIGURE, BlockType.FOOTNOTE}


def render_chapter_text(chapter: Chapter) -> str:
    parts: list[str] = []
    for block in chapter.blocks:
        if block.type in _SKIPPED:
            continue
        text = normalize(block.text)
        if not text:
            continue
        if block.type is BlockType.HEADING:
            parts.append(f"{text.rstrip('.')}. {PAUSE}")
        else:
            parts.append(text)
    return "\n\n".join(parts)
