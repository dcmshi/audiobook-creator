import pytest

from audiobook_creator.process.rules import normalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Results were strong [12].", "Results were strong."),
        ("As shown [3, 4] and [7-9].", "As shown and."),
        ("See Fig. 3 for details.", "See Figure 3 for details."),
        ("Per Eq. 2 above.", "Per Equation 2 above."),
        ("Smith et al. found this.", "Smith and colleagues found this."),
        ("Fruits, e.g. apples, are good.", "Fruits, for example, apples, are good."),
        ("The limit, i.e. the cap.", "The limit, that is, the cap."),
        ("Cats vs. dogs.", "Cats versus dogs."),
        ("Growth of 40% overall.", "Growth of 40 percent overall."),
        ("R&D spending rose.", "R and D spending rose."),
        ("Too   many    spaces.", "Too many spaces."),
    ],
)
def test_normalize(raw: str, expected: str):
    assert normalize(raw) == expected


def test_normalize_leaves_plain_prose_alone():
    text = "It was a dark and stormy night."
    assert normalize(text) == text
