from audiobook_creator.models import BlockType, Chapter
from audiobook_creator.process.rules import normalize

PAUSE = "[[pause]]"

# Must track the placeholder title split_chapters() gives blocks before the first heading.
_PLACEHOLDER_TITLE = "Beginning"

# Verbatim v1 contract: read prose and captions; skip tables, figures, and
# footnote bodies (their inline markers are stripped by normalize()).
_SKIPPED = {BlockType.TABLE, BlockType.FIGURE, BlockType.FOOTNOTE}


def render_chapter_text(chapter: Chapter) -> str:
    parts: list[str] = []
    # split_chapters() consumes the boundary heading into Chapter.title, so the title
    # is announced here or never heard. "Beginning" is the synthetic placeholder for
    # blocks preceding the first heading and must not be spoken.
    if chapter.title != _PLACEHOLDER_TITLE:
        title = normalize(chapter.title)
        if title:
            parts.append(f"{title.rstrip('.')}. {PAUSE}")
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
