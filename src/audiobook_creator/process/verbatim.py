from collections.abc import Callable

from audiobook_creator.models import BlockType, Chapter
from audiobook_creator.process.rules import normalize

PAUSE = "[[pause]]"

# Must track the placeholder title split_chapters() gives blocks before the first heading.
PLACEHOLDER_TITLE = "Beginning"

# Verbatim v1 contract: read prose and captions; skip tables, figures, and
# footnote bodies (their inline markers are stripped by normalize()).
_SKIPPED = {BlockType.TABLE, BlockType.FIGURE, BlockType.FOOTNOTE}

# Rewrite mode speaks tables and figures; footnote bodies stay out of the narration there too.
_SKIPPED_VISUAL = {BlockType.FOOTNOTE}


def has_speakable_blocks(chapter: Chapter, include_visual: bool = False) -> bool:
    """True when blocks contribute narration beyond the chapter's title announcement.

    `include_visual` is for rewrite mode, which verbalizes tables and figures instead of
    skipping them — there, a chapter of nothing but tables is content, not silence.
    """
    skipped = _SKIPPED_VISUAL if include_visual else _SKIPPED
    return any(block.type not in skipped and normalize(block.text) for block in chapter.blocks)


def render_chapter_text(chapter: Chapter, normalizer: Callable[[str], str] = normalize) -> str:
    parts: list[str] = []
    # split_chapters() consumes the boundary heading into Chapter.title, so the title
    # is announced here or never heard. "Beginning" is the synthetic placeholder for
    # blocks preceding the first heading and must not be spoken.
    if chapter.title != PLACEHOLDER_TITLE:
        # The title line stays on the rule path: it is short, and the announcement frame
        # around it must stay deterministic.
        title = normalize(chapter.title)
        if title:
            parts.append(f"{title.rstrip('.')}. {PAUSE}")
    for block in chapter.blocks:
        if block.type in _SKIPPED:
            continue
        text = normalizer(block.text)
        if not text:
            continue
        if block.type is BlockType.HEADING:
            parts.append(f"{text.rstrip('.')}. {PAUSE}")
        else:
            parts.append(text)
    return "\n\n".join(parts)
