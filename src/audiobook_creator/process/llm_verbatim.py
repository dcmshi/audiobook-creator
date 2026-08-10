import logging
from collections.abc import Callable
from pathlib import Path

from audiobook_creator.process.llm.base import LLMClient, LLMError
from audiobook_creator.process.llm.cache import cached_complete
from audiobook_creator.process.rules import normalize

logger = logging.getLogger(__name__)

# Byte-stable: this is the cache key prefix, so editing it invalidates every cached response.
VERBATIM_SYSTEM = """You normalize text for text-to-speech narration. Rewrite the user's text so it reads
aloud naturally, changing NOTHING else:
- Expand abbreviations and units ("Dr." -> "Doctor", "km/h" -> "kilometers per hour").
- Write numbers, dates, currencies, percentages, and equations as they should be spoken.
- Remove citation markers like [12], (Smith et al., 2020) footnote marks, and URLs;
  when a citation names authors, keep a natural attribution ("Smith and colleagues").
- Never summarize, shorten, reorder, add, or omit content. Keep every sentence.
- Output plain prose only: no markdown, no headings, no lists, no XML, no quotes around
  the whole text, no commentary about what you did.
Reply with the normalized text and nothing else."""

_FORBIDDEN = ("#", "<", "**", "[[")


def _valid(source: str, out: str) -> bool:
    out = out.strip()
    if not out:
        return False
    # A model that summarizes or pads is not normalizing; either way the rule path is safer
    # than shipping content the book does not have.
    ratio = len(out) / max(len(source), 1)
    if not 0.5 <= ratio <= 1.5:
        return False
    return not any(marker in out for marker in _FORBIDDEN)


def make_llm_normalizer(client: LLMClient, cache_dir: Path) -> Callable[[str], str]:
    def normalizer(text: str) -> str:
        try:
            out = cached_complete(client, cache_dir, text, system=VERBATIM_SYSTEM)
        except LLMError as exc:
            logger.warning("LLM normalization failed, falling back to rules: %s", exc)
            return normalize(text)
        if not _valid(text, out):
            logger.warning("LLM normalization rejected by validator, falling back to rules")
            return normalize(text)
        return " ".join(out.split())

    return normalizer
