import hashlib
import logging
import re
from pathlib import Path

from audiobook_creator.models import Block, BlockType, Chapter
from audiobook_creator.process.llm.base import LLMClient
from audiobook_creator.process.llm.cache import _cache_key, cached_complete
from audiobook_creator.process.rules import normalize
from audiobook_creator.process.verbatim import PAUSE, PLACEHOLDER_TITLE

logger = logging.getLogger(__name__)

# Byte-stable: both prompts are cache key prefixes, so editing either invalidates the cache.
REWRITE_SYSTEM = """You rewrite document text so it can be listened to as an audiobook. Preserve all
substantive content, arguments, findings, and numbers, but re-express anything that
only works on paper:
- [TABLE] blocks: replace with a fluent spoken summary of what the table shows,
  reading out the key figures. Never read cells row by row.
- [FIGURE ...] blocks: the description provided is what a sighted reader would see;
  weave it in naturally ("The figure shows...").
- Inline citations become natural attributions or are dropped; equations are spoken
  ("x squared plus y") or described intuitively.
- Keep the author's voice and order. Do not condense beyond what listening requires,
  do not editorialize, do not add introductions or conclusions.
Output plain prose only. You may keep [[pause]] markers where a natural break falls.
No markdown, no headings, no lists, no commentary about your changes."""

FIGURE_PROMPT = (
    "Describe this figure for an audiobook listener in 2-4 sentences: what it shows, "
    "the axes or categories if any, and the one takeaway a sighted reader would get. "
    "Plain prose only."
)

_WINDOW_LIMIT = 6000
_WINDOW_MAX_TOKENS = 8192

# [[pause]] is the one marker the prompt allows through; everything else is scaffolding the
# TTS would read aloud. Parked behind a sentinel first so cleanup order cannot damage it.
_PAUSE_SENTINEL = "\x00pause\x00"
# [TABLE] / [FIGURE fig-000] are input-side annotations. If the model echoes one, the prose it
# annotated is already there, so the marker is pure noise. Matches one or two brackets.
_INPUT_MARKER = re.compile(r"\[\[?(?:TABLE|FIGURE)\b[^\]]*\]\]?")
_STRAY_BRACKETS = re.compile(r"\[\[|\]\]")


def _strip_markers(text: str) -> str:
    text = text.replace(PAUSE, _PAUSE_SENTINEL)
    text = _INPUT_MARKER.sub("", text)
    text = _STRAY_BRACKETS.sub("", text)
    return text.replace(_PAUSE_SENTINEL, PAUSE)


def _normalize_prose(text: str) -> str:
    """Apply the rule pass the model's output never went through.

    Per paragraph, because normalize() collapses every run of whitespace — over a whole
    window it would weld the model's paragraphs into one block.
    """
    paragraphs = (normalize(part) for part in text.split("\n\n"))
    return "\n\n".join(part for part in paragraphs if part)


def _cached_describe(client: LLMClient, cache_dir: Path, image_path: Path, caption: str) -> str:
    """Vision description for one figure, cached on the image bytes rather than its path.

    cached_complete cannot wrap vision, so this mirrors its key encoding and atomic write.
    Keying on content means a re-ingest that renumbers fig-NNN still hits the cache.
    """
    digest = hashlib.sha1(image_path.read_bytes()).hexdigest()
    model = getattr(client, "model", "")
    key = _cache_key([client.name, model, digest, caption])
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"img-{key}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    text = client.describe_image(image_path, FIGURE_PROMPT)
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    return text


def _describe_figure(block: Block, client: LLMClient, cache_dir: Path) -> str:
    """The figure's spoken text: its description when vision works, else its caption."""
    if not block.image_path:
        return block.text
    try:
        described = _cached_describe(client, cache_dir, Path(block.image_path), block.text)
    # Broad on purpose: this runs outside the per-window try, so an unanticipated provider
    # exception here would cost the whole book rather than one figure its description.
    # LLMUnsupported subclasses LLMError, so a vision-less provider lands here too.
    except Exception as exc:  # noqa: BLE001
        logger.info("figure description unavailable, using caption: %s", exc)
        return block.text
    return described.strip() or block.text


def _resolve_figures(chapter: Chapter, client: LLMClient, cache_dir: Path) -> list[str]:
    """Spoken text per block, aligned 1:1 with chapter.blocks ('' = nothing to narrate)."""
    spoken: list[str] = []
    for block in chapter.blocks:
        if block.type is BlockType.FOOTNOTE:
            # Footnote bodies are not narrated in verbatim either; their inline markers are
            # stripped from the prose that cites them.
            spoken.append("")
        elif block.type is BlockType.FIGURE:
            spoken.append(_describe_figure(block, client, cache_dir))
        else:
            spoken.append(block.text)
    return spoken


def _serialize(block: Block, spoken: str, index: int) -> str:
    """Mark up a block so the rewrite prompt can tell prose from tables and figures."""
    if block.type is BlockType.TABLE:
        return f"[TABLE] {spoken}"
    if block.type is BlockType.FIGURE:
        fig_id = Path(block.image_path).stem if block.image_path else f"fig-{index:03d}"
        return f"[FIGURE {fig_id}] {spoken}"
    return spoken


def _windows(texts: list[str], limit: int = _WINDOW_LIMIT) -> list[list[int]]:
    """Group indices so each window's joined text stays under `limit` characters.

    A single block longer than the limit becomes its own oversized window: splitting mid
    block would hand the model a fragment, and the provider's own limits are the backstop.
    """
    windows: list[list[int]] = []
    current: list[int] = []
    size = 0
    for index, text in enumerate(texts):
        cost = len(text) + 2  # the "\n\n" that will join it
        if current and size + cost > limit:
            windows.append(current)
            current, size = [], 0
        current.append(index)
        size += cost
    if current:
        windows.append(current)
    return windows


# Same output contract the verbatim validator enforces: prose only, nothing a narrator would
# have to read as a symbol.
_FORBIDDEN_MARKUP = ("#", "**", "<")


def _is_speakable(text: str) -> bool:
    if any(marker in text for marker in _FORBIDDEN_MARKUP):
        return False
    # [[pause]] is the one bracket pair the prompt allows; anything else survived cleanup.
    return "[[" not in text.replace(PAUSE, "")


# Markdown and XML in their real forms. Deliberately NOT part of _strip_markers: that runs
# before _is_speakable, so sanitising there would leave the validator nothing to catch and
# silently pass markup-bearing rewrites through.
#
# Both patterns are bounded to a single line, because this runs over blocks already joined and
# it is the one path whose job is never to lose content:
#   - a tag needs a tag-like opening and no newline, or prose comparing "a < b" in one block to
#     "c > d" in a later one would have everything between them deleted (get_text() decodes
#     &lt;/&gt; back into literal angle brackets, so that prose really does reach block text);
#   - the heading marker uses [ \t] rather than \s, or the blank line before "## B" is eaten and
#     a paragraph break degrades to a line break that _normalize_prose cannot restore.
_MARKUP = re.compile(r"<[/a-zA-Z][^>\n]*>|\*\*|^[ \t]*#{1,6}[ \t]*", re.MULTILINE)


def _verbatim_window(items: list[tuple[Block, str, str]], window: list[int]) -> str:
    """The window's own blocks, cleaned and rule-normalized — the fallback for a bad rewrite.

    Markup is stripped here rather than rejected: the fallback is the last resort, so there
    is nothing further to fall back to. It needs the cleanup because it is not purely source
    text — a FIGURE's spoken text is its vision description, model output that passes through
    no other cleanup.
    """
    joined = "\n\n".join(items[i][2] for i in window)
    return _normalize_prose(_MARKUP.sub("", _strip_markers(joined)))


def render_rewrite(chapter: Chapter, client: LLMClient, cache_dir: Path) -> str:
    spoken = _resolve_figures(chapter, client, cache_dir)
    items = [
        (block, _serialize(block, text, index), text)
        for index, (block, text) in enumerate(zip(chapter.blocks, spoken, strict=True))
        if text.strip()
    ]
    parts: list[str] = []
    # split_chapters() folds the boundary heading into Chapter.title, so it is announced
    # here or never heard. The frame stays deterministic: rules only, never the LLM.
    if chapter.title != PLACEHOLDER_TITLE:
        title = normalize(chapter.title)
        if title:
            parts.append(f"{title.rstrip('.')}. {PAUSE}")

    serialized = [text for _block, text, _spoken in items]
    for window in _windows(serialized):
        prompt = "\n\n".join(serialized[i] for i in window)
        try:
            rewritten = cached_complete(
                client,
                cache_dir,
                prompt,
                system=REWRITE_SYSTEM,
                max_tokens=_WINDOW_MAX_TOKENS,
            )
        # Broad on purpose, per the verbatim-normalizer precedent: this is the degradation
        # point, and a malformed provider response must cost one window its rewrite rather
        # than cost the book its conversion.
        except Exception as exc:  # noqa: BLE001
            logger.warning("window rewrite failed, falling back to verbatim text: %s", exc)
            rewritten = ""
        else:
            # Markers first, then the rule pass: stripping leaves stray whitespace that
            # normalize() tidies, and normalize() would otherwise run over scaffolding text.
            rewritten = _normalize_prose(_strip_markers(rewritten))
            # Each of the three roads to the fallback warns exactly once.
            if not rewritten:
                logger.warning("window rewrite returned no text, falling back to verbatim text")
            elif not _is_speakable(rewritten):
                logger.warning("window rewrite contained markup, falling back to verbatim text")
                rewritten = ""
        if not rewritten:
            rewritten = _verbatim_window(items, window)
        if rewritten:
            parts.append(rewritten)
    return "\n\n".join(parts)
