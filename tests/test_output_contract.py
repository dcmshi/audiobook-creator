import pytest

from audiobook_creator.process.output_contract import PAUSE, is_speakable, sanitize

_REJECTED = [
    ("heading", "## Results"),
    ("hash anywhere", "Section #4 follows."),
    ("bold", "Growth was **strong**."),
    ("html tag", "See <em>panel A</em>."),
    ("bare angle", "The guard holds if a < b."),
    ("code span", "Call `run()` first."),
    ("dash bullet", "Findings:\n- first item"),
    ("star bullet", "Findings:\n* first item"),
    ("indented bullet", "Findings:\n  - first item"),
    ("blockquote", "As noted:\n> quoted text"),
    ("stray double bracket", "Growth [[TABLE]] here."),
    ("pause does not launder", f"{PAUSE} ## Heading"),
]

_ACCEPTED = [
    ("plain prose", "Growth reached forty percent by 2026."),
    ("legal pause", f"Growth was strong. {PAUSE} The figure shows a rise."),
    ("hyphen mid-sentence", "A well-known result - widely cited - holds here."),
    ("star mid-sentence", "The 5*3 grid was measured."),
    ("greater-than mid-sentence", "Values above 5 > 3 were kept."),
]


@pytest.mark.parametrize(("label", "text"), _REJECTED, ids=[label for label, _ in _REJECTED])
def test_markup_is_rejected(label, text):
    assert is_speakable(text) is False


@pytest.mark.parametrize(("label", "text"), _ACCEPTED, ids=[label for label, _ in _ACCEPTED])
def test_prose_is_accepted(label, text):
    assert is_speakable(text) is True


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("## Results heading", "Results heading"),
        ("Growth was **strong**.", "Growth was strong."),
        ("See <em>panel A</em>.", "See panel A."),
        ("Call `run()` first.", "Call run() first."),
        ("Findings:\n- first item", "Findings:\nfirst item"),
        ("Findings:\n* first item", "Findings:\nfirst item"),
        ("As noted:\n> quoted text", "As noted:\nquoted text"),
        ("Growth [[TABLE]] here.", "Growth TABLE here."),
    ],
)
def test_sanitize_removes_markup(text, expected):
    assert sanitize(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "The guard holds if a < b in every case.\n\nIt fails when c > d here.",
        "A\n\n## B",
        f"Growth was strong. {PAUSE} The figure shows a rise.",
        "The 5*3 grid was measured.",
    ],
)
def test_sanitize_leaves_prose_and_structure_intact(text):
    """The removal pass runs where content cannot be recovered, so it must not overreach."""
    expected = text.replace("## ", "")  # only the heading marker should ever go
    assert sanitize(text) == expected


# Markup in its canonical written form: sanitize must be able to clear these completely.
_CANONICAL_MARKUP = [
    "## Results heading",
    "Growth was **strong**.",
    "See <em>panel A</em>.",
    "Call `run()` first.",
    "Findings:\n- first item",
    "Findings:\n* first item",
    "As noted:\n> quoted text",
    "Growth [[TABLE]] here.",
]


@pytest.mark.parametrize("text", _CANONICAL_MARKUP)
def test_sanitized_canonical_markup_becomes_speakable(text):
    assert is_speakable(text) is False
    assert is_speakable(sanitize(text)) is True


@pytest.mark.parametrize("text", ["Section #4 follows.", "The guard holds if a < b."])
def test_bare_symbols_are_rejected_but_never_deleted(text):
    """The halves are asymmetric on purpose.

    A lone `#` or `<` in prose is not markup, so the removal pass leaves it: this pass runs
    where content cannot be recovered, and deleting a real character to satisfy a checker
    would be the worse failure. The rejection side stays strict because it only costs a
    fallback, so these still fail is_speakable and route to rules.
    """
    assert is_speakable(text) is False
    assert sanitize(text) == text
