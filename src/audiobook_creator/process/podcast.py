import logging
import re
from pathlib import Path

from audiobook_creator.models import Chapter
from audiobook_creator.process.llm.base import LLMClient, LLMError
from audiobook_creator.process.llm.cache import cached_complete
from audiobook_creator.process.rules import normalize

logger = logging.getLogger(__name__)

# Byte-stable: this is the cache key prefix, so editing it invalidates every cached script.
PODCAST_SYSTEM = """You turn a document into a two-host podcast conversation. Host 1 guides and asks;
Host 2 explains, with concrete numbers and examples from the text. Cover every major
section and finding of the document, front to back, in 10 to 20 minutes of speech.
Format: one line per utterance, each line starting with [[speaker:1]] or [[speaker:2]].
Plain spoken prose after the tag - no markdown, no stage directions, no names, no intro
music cues. Start with Host 1 introducing the work in one sentence."""

_SOURCE_CAP = 300_000
_SCRIPT_MAX_TOKENS = 16000
_MIN_UTTERANCES = 4

# The one marker this mode emits on purpose: Task 10's synthesis reads it to pick a voice.
_SPEAKER_LINE = re.compile(r"^\[\[speaker:[12]\]\]")


def _assemble_source(chapters: list[Chapter]) -> str:
    """Body text for the prompt: chapter headers plus rule-normalized block text."""
    parts: list[str] = []
    for chapter in chapters:
        parts.append(f"Chapter: {chapter.title}")
        # Tables and figures contribute their own text; the host is expected to talk about
        # what they contain, so nothing is skipped here.
        parts.extend(text for block in chapter.blocks if (text := normalize(block.text)))
    return "\n\n".join(parts)


def _cap_source(source: str) -> str:
    if len(source) <= _SOURCE_CAP:
        return source
    kept = source[:_SOURCE_CAP]
    logger.warning(
        "document is %d characters; the podcast source is truncated at %d, dropping %d. "
        "The last text kept is %r, and everything after it is not covered by the script.",
        len(source),
        _SOURCE_CAP,
        len(source) - _SOURCE_CAP,
        kept[-60:],
    )
    return kept


def _validate_script(script: str) -> str:
    """Normalize the model's script into one utterance per line, or refuse it.

    Podcast has no rule-based equivalent, so an unusable script fails loudly instead of
    degrading — a silently half-formed dialogue is worse than a job that stops.
    """
    utterances: list[str] = []
    for raw in script.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _SPEAKER_LINE.match(line):
            utterances.append(line)
        elif utterances:
            # A wrapped utterance, not a new one: rejoin it rather than losing the text.
            utterances[-1] = f"{utterances[-1]} {line}"
        else:
            logger.debug("dropping preamble before the first speaker tag: %r", line)
    if len(utterances) < _MIN_UTTERANCES:
        raise LLMError(
            f"podcast script had {len(utterances)} speaker lines, expected at least "
            f"{_MIN_UTTERANCES} starting with [[speaker:1]] or [[speaker:2]]"
        )
    return "\n".join(utterances)


def render_podcast(
    doc_title: str, chapters: list[Chapter], client: LLMClient, cache_dir: Path
) -> str:
    # Cap the assembled prompt, not just the body: the cap exists to bound what is sent, and
    # the title header would otherwise push the request past it.
    user = _cap_source(f"Document title: {doc_title}\n\n{_assemble_source(chapters)}")
    # No try/except: an LLMError here must reach the stage. See _validate_script.
    script = cached_complete(
        client, cache_dir, user, system=PODCAST_SYSTEM, max_tokens=_SCRIPT_MAX_TOKENS
    )
    return _validate_script(script)
