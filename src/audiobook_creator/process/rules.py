import re

# "[12]", "[3, 4]", "[7-9]", "[7–9]" — bracketed numeric citation markers
_CITATION = re.compile(r"\s?\[\d+(?:\s*[,–-]\s*\d+)*\]")

_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bFig\.\s*"), "Figure "),
    (re.compile(r"\bEq\.\s*"), "Equation "),
    (re.compile(r"\bet al\.", re.IGNORECASE), "and colleagues"),
    (re.compile(r"\be\.g\.,?\s*", re.IGNORECASE), "for example, "),
    (re.compile(r"\bi\.e\.,?\s*", re.IGNORECASE), "that is, "),
    (re.compile(r"\bvs\.\s*", re.IGNORECASE), "versus "),
    (re.compile(r"\s*%"), " percent"),
    (re.compile(r"\s*&\s*"), " and "),
]


def normalize(text: str) -> str:
    text = _CITATION.sub("", text)
    for pattern, replacement in _REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()
