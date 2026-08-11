import logging
import re
from pathlib import Path

from audiobook_creator.models import Chapter
from audiobook_creator.process.llm.base import LLMClient, LLMError
from audiobook_creator.process.llm.cache import cached_complete, evict
from audiobook_creator.process.output_contract import is_speakable
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
_SPEAKER_LINE = re.compile(r"^\[\[speaker:([12])\]\]")


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

    cleaned: list[str] = []
    speakers: set[str] = set()
    for utterance in utterances:
        match = _SPEAKER_LINE.match(utterance)
        tag, body = match.group(0).strip(), utterance[match.end() :]
        speakers.add(match.group(1))
        # Rejection rather than sanitising, unlike verbatim and rewrite: this mode has no
        # rule-based path to fall back to, so a bad script is a failure, not a degradation.
        if not is_speakable(body):
            raise LLMError(f"podcast utterance contains markup: {utterance[:80]!r}")
        # The tag is this mode's contract with synthesis; only the spoken part is normalized.
        cleaned.append(f"{tag} {normalize(body)}".strip())
    if not {"1", "2"} <= speakers:
        raise LLMError(
            "podcast script did not use both hosts; expected [[speaker:1]] and [[speaker:2]]"
        )
    return "\n".join(cleaned)


def render_podcast(
    doc_title: str, chapters: list[Chapter], client: LLMClient, cache_dir: Path
) -> str:
    # Cap the assembled prompt, not just the body: the cap exists to bound what is sent, and
    # the title header would otherwise push the request past it.
    user = _cap_source(f"Document title: {doc_title}\n\n{_assemble_source(chapters)}")
    # No try/except around the call itself: an LLMError from the provider must reach the stage.
    script = cached_complete(
        client, cache_dir, user, system=PODCAST_SYSTEM, max_tokens=_SCRIPT_MAX_TOKENS
    )
    try:
        return _validate_script(script)
    except LLMError:
        # The key is deterministic, so a rejected script left in the cache would be re-read and
        # re-rejected on every later run — the job would be permanently unrunnable, and nothing
        # in the error would point at the cache. Drop it so the next run regenerates.
        evict(client, cache_dir, user, system=PODCAST_SYSTEM, max_tokens=_SCRIPT_MAX_TOKENS)
        raise
