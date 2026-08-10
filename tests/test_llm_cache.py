from pathlib import Path

import pytest

from audiobook_creator.process.llm.base import LLMError
from audiobook_creator.process.llm.cache import cached_complete


class CountingLLM:
    name = "fake"
    model = "m1"

    def __init__(self, fail_first=False):
        self.calls = 0
        self.fail_first = fail_first

    def complete(self, user, *, system=None, max_tokens=2048):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise LLMError("transient")
        return f"result:{user}"

    def describe_image(self, image_path, prompt, *, max_tokens=1024):
        return "img"


def test_hit_skips_second_call(tmp_path: Path):
    llm = CountingLLM()
    a = cached_complete(llm, tmp_path, "chunk one", system="s")
    b = cached_complete(llm, tmp_path, "chunk one", system="s")
    assert a == b == "result:chunk one"
    assert llm.calls == 1


def test_different_system_is_different_key(tmp_path: Path):
    llm = CountingLLM()
    cached_complete(llm, tmp_path, "chunk", system="s1")
    cached_complete(llm, tmp_path, "chunk", system="s2")
    assert llm.calls == 2


def test_failures_are_not_cached(tmp_path: Path):
    llm = CountingLLM(fail_first=True)
    with pytest.raises(LLMError):
        cached_complete(llm, tmp_path, "chunk")
    assert list(tmp_path.glob("*.txt")) == []
    assert cached_complete(llm, tmp_path, "chunk") == "result:chunk"


def test_delimiter_shift_is_not_a_collision(tmp_path: Path):
    """A bare "|" join hashes these two pairs identically, serving one the other's text."""
    llm = CountingLLM()
    a = cached_complete(llm, tmp_path, "c", system="a|b")
    b = cached_complete(llm, tmp_path, "b|c", system="a")
    assert llm.calls == 2
    assert a == "result:c"
    assert b == "result:b|c"


def test_max_tokens_is_part_of_the_key(tmp_path: Path):
    llm = CountingLLM()
    cached_complete(llm, tmp_path, "chunk", max_tokens=64)
    cached_complete(llm, tmp_path, "chunk", max_tokens=4096)
    assert llm.calls == 2


def test_successful_write_leaves_no_temp_file(tmp_path: Path):
    llm = CountingLLM()
    assert cached_complete(llm, tmp_path, "chunk") == "result:chunk"
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_interrupted_write_leaves_nothing_servable(tmp_path: Path, monkeypatch):
    """A half-written file must not become a permanent cache hit."""
    llm = CountingLLM()

    def killed_mid_write(self, target):
        raise OSError("interrupted")

    monkeypatch.setattr(Path, "replace", killed_mid_write)
    with pytest.raises(OSError):
        cached_complete(llm, tmp_path, "chunk")
    assert list(tmp_path.glob("*.txt")) == []
    assert list(tmp_path.iterdir()) == []  # the aside file is cleaned up too
    monkeypatch.undo()
    assert cached_complete(llm, tmp_path, "chunk") == "result:chunk"
    assert llm.calls == 2
