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
