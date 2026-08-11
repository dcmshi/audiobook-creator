import logging
import re

from audiobook_creator.models import Block, BlockType, Chapter, Document, Matter

logger = logging.getLogger(__name__)

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


TIEBREAK_PROMPT = (
    "Classify each chapter title as front (title pages, contents, dedications), "
    "body (the actual content), or back (references, index, appendices). "
    'Reply with one line per chapter: "<index>: front|body|back". Titles:\n'
)

_MATTER_WORDS = {"front": Matter.FRONT, "body": Matter.BODY, "back": Matter.BACK}
_LABEL_LINE = re.compile(r"^\s*(\d+)\s*:\s*(front|body|back)\s*$", re.IGNORECASE)

# Front matter runs at the start and back matter at the end; a chapter in the middle being
# relabelled away from BODY is how a real chapter goes silent, so only the edges may do it.
_FRONT_EDGE = 2
_BACK_EDGE = 3


def _allowed(index: int, total: int, current: Matter, proposed: Matter) -> bool:
    if proposed is current:
        return False
    if proposed is Matter.BODY:
        return True  # rescuing content is always safe: the worst case is narrating too much
    # The midpoint guard matters only for short books, where a fixed window would otherwise
    # cover the whole document: in a four-chapter book "the last three" reaches index 1.
    midpoint = total / 2
    if proposed is Matter.FRONT:
        return index < _FRONT_EDGE and index < midpoint
    return index >= total - _BACK_EDGE and index > midpoint


def refine_matter_with_llm(chapters: list[Chapter], client) -> list[Chapter]:
    """Let an LLM re-label front/body/back, gated so it can never silence a real chapter.

    Any failure — a client error, an unparsable reply, a label for an unknown index —
    leaves the rule-based classification untouched.
    """
    if not chapters:
        return chapters
    titles = "\n".join(f"{c.index}. {c.title}" for c in chapters)
    try:
        reply = client.complete(TIEBREAK_PROMPT + titles)
    except Exception as exc:  # noqa: BLE001 - a tiebreaker must never fail the stage
        logger.warning("chapter tiebreaker unavailable, keeping rule-based matter: %s", exc)
        return chapters

    proposed: dict[int, Matter] = {}
    for line in reply.splitlines():
        match = _LABEL_LINE.match(line)
        if match:
            proposed[int(match.group(1))] = _MATTER_WORDS[match.group(2).lower()]
    if not proposed:
        logger.warning("chapter tiebreaker reply had no usable labels, keeping rule-based matter")
        return chapters

    by_index = {c.index: c for c in chapters}
    total = len(chapters)
    for position, chapter in enumerate(chapters):
        label = proposed.get(chapter.index)
        if label is None or chapter.index not in by_index:
            continue
        if _allowed(position, total, chapter.matter, label):
            logger.info(
                "tiebreaker: chapter %d %r %s -> %s",
                chapter.index, chapter.title, chapter.matter.value, label.value,
            )
            chapter.matter = label
        elif label is not chapter.matter:
            logger.info(
                "tiebreaker: rejected %r for mid-document chapter %d %r",
                label.value, chapter.index, chapter.title,
            )
    return chapters
