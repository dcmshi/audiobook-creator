"""What may reach processed/*.txt: speakable prose, and nothing a narrator reads as a symbol.

Two halves of one contract, and the difference matters. `is_speakable` is a *rejection* test,
used where a fallback exists — it can afford to be strict, because a false positive costs one
window its LLM pass. `sanitize` is a *removal* pass, used where nothing is left to fall back
to, so it is bounded to markup shapes and must never eat ordinary prose.
"""

import re

PAUSE = "[[pause]]"

# Markers that would be narrated as punctuation. `<` is rejected outright rather than matched
# as a tag: on the rejection side a false positive is cheap, and the fallback is prose either way.
_FORBIDDEN_SUBSTRINGS = ("#", "**", "<", "`")
_FORBIDDEN_LINE_STARTS = re.compile(r"^[ \t]*(?:[-*][ \t]|>[ \t])", re.MULTILINE)


def is_speakable(text: str) -> bool:
    """True when nothing in `text` would be read aloud as markup."""
    if any(marker in text for marker in _FORBIDDEN_SUBSTRINGS):
        return False
    if _FORBIDDEN_LINE_STARTS.search(text):
        return False
    # [[pause]] is the one bracket pair the prompts allow through.
    return "[[" not in text.replace(PAUSE, "")


# Every alternative is bounded to a single line and to a markup-like shape. A bare `<` is left
# alone here: prose comparing "a < b" in one block to "c > d" in a later one would otherwise
# have everything between them deleted, and this pass runs where content cannot be recovered.
_MARKUP = re.compile(
    r"<[/a-zA-Z][^>\n]*>"  # html/xml tags
    r"|\*\*"  # bold
    r"|`+"  # code spans
    r"|^[ \t]*#{1,6}[ \t]*"  # headings
    r"|^[ \t]*[-*][ \t]+"  # list bullets
    r"|^[ \t]*>[ \t]+",  # blockquotes
    re.MULTILINE,
)
_PAUSE_SENTINEL = "\x00pause\x00"
_STRAY_BRACKETS = re.compile(r"\[\[|\]\]")


def sanitize(text: str) -> str:
    """Strip what `is_speakable` rejects, leaving ordinary prose untouched."""
    text = text.replace(PAUSE, _PAUSE_SENTINEL)
    text = _MARKUP.sub("", text)
    text = _STRAY_BRACKETS.sub("", text)
    return text.replace(_PAUSE_SENTINEL, PAUSE)
