# Core Pipeline + CLI Implementation Plan (Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working CLI (`abc`) that converts an EPUB or PDF into per-chapter MP3s and/or an M4B audiobook with chapter markers, using rule-based verbatim normalization and local Kokoro TTS, with resumable file-based jobs.

**Architecture:** Five idempotent pipeline stages (ingest → structure → process → synthesize → package) communicating via files in `jobs/<id>/`. Each stage is a `run_stage(job)` function; the engine runs stages in order, skipping completed ones. EPUB is parsed natively (zipfile + BeautifulSoup); PDF/DOCX/HTML go through Docling behind an optional extra.

**Tech Stack:** Python 3.12+, uv, pydantic v2, Typer, BeautifulSoup4, kokoro-onnx + numpy, ffmpeg (external binary), pytest. Docling via `[pdf]` extra only.

**Plan sequence:** Plan 1 of 3. Plan 2 adds the LLM layer (API/Ollama clients, LLM verbatim, rewrite, podcast modes). Plan 3 adds the web UI. Spec: `docs/superpowers/specs/2026-08-06-audiobook-creator-design.md`.

## Global Constraints

- Python `>=3.12`; environment managed with `uv`; `src/` layout; package `audiobook_creator`; CLI entry point named `abc`.
- Base install MUST NOT require PyTorch/CUDA. Docling (which pulls torch) lives behind the `[pdf]` optional extra and is imported lazily with an actionable error message.
- No AGPL dependencies (specifically: no `ebooklib`, no `pymupdf`/`pymupdf4llm`).
- Every `open()`/`read_text()`/`write_text()` MUST pass `encoding="utf-8"` (Windows defaults to cp1252).
- Every `subprocess.run` MUST use list-form args (never `shell=True`), `check=True`, `capture_output=True`.
- Use `pathlib.Path` everywhere; paths written into ffmpeg concat lists use `.as_posix()`.
- Tests are hermetic: no network, no model weights, no API keys. Tests needing ffmpeg/ffprobe auto-skip when the binaries are absent. Tests needing docling or kokoro models are marked and skipped by default.
- Commit message prefixes: `feat:`, `test:`, `fix:`, `chore:`.
- Primary dev platform is Windows 11 — no POSIX-only assumptions (no `os.fork`, no `/tmp`, no `chmod` reliance).

## File Structure

```
pyproject.toml
src/audiobook_creator/
├── __init__.py
├── models.py                  # pydantic models + STAGES constant
├── core/
│   ├── __init__.py
│   ├── job.py                 # Job: dir layout, create/load/save
│   └── engine.py              # run(job, from_stage): ordered, skip-done, record failures
├── ingest/
│   ├── __init__.py
│   ├── epub.py                # ingest_epub(path, assets_dir) -> Document
│   ├── docling_adapter.py     # document_from_docling(dl_doc) -> Document; ingest_with_docling(...)
│   └── stage.py               # dispatcher by suffix/URL + run_stage(job)
├── structure/
│   ├── __init__.py
│   ├── chapters.py            # split_chapters(doc) -> list[Chapter]; classify_matter(...)
│   └── stage.py
├── process/
│   ├── __init__.py
│   ├── rules.py               # normalize(text) -> str  (rule-based, no LLM)
│   ├── verbatim.py            # render_chapter_text(chapter) -> str
│   └── stage.py
├── synthesize/
│   ├── __init__.py
│   ├── base.py                # TTSBackend protocol, registry, PrivacyError, chunk_text, write_wav
│   ├── stub.py                # StubBackend (440 Hz tone)
│   ├── kokoro.py              # KokoroBackend (kokoro-onnx)
│   └── stage.py               # chunk cache, retry→silence, per-chapter WAVs
├── package/
│   ├── __init__.py
│   ├── ffmpeg.py              # mp3 encode, ffmetadata, m4b mux, ffprobe helpers
│   └── stage.py
└── cli.py                     # Typer app: convert, jobs, resume, preview, doctor
tests/
├── conftest.py                # fixture EPUB builder, ffmpeg availability marker
├── test_models.py
├── test_job.py
├── test_engine.py
├── test_rules.py
├── test_epub_ingest.py
├── test_docling_adapter.py
├── test_structure.py
├── test_verbatim.py
├── test_tts_base.py
├── test_synthesize_stage.py
├── test_package_stage.py
├── test_cli.py
└── test_kokoro.py             # marked, skipped without model files
```

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `src/audiobook_creator/__init__.py`, `tests/test_smoke.py`, `README.md`

**Interfaces:**
- Produces: importable `audiobook_creator` package with `__version__`; `uv run pytest` works.

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "audiobook-creator"
version = "0.1.0"
description = "Turn PDFs and EPUBs into audiobooks (MP3/M4B) with local-first TTS"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.7",
    "typer>=0.12",
    "beautifulsoup4>=4.12",
    "numpy>=1.26",
    "kokoro-onnx>=0.4",
]

[project.optional-dependencies]
pdf = ["docling>=2.0"]

[project.scripts]
abc = "audiobook_creator.cli:app"

[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.5"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/audiobook_creator"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "docling: requires docling installed and a sample PDF (skipped by default)",
    "kokoro: requires kokoro model files under models/ (skipped by default)",
]
addopts = "-m 'not docling and not kokoro'"

[tool.ruff]
line-length = 100
target-version = "py312"
```

- [ ] **Step 2: Write package init and smoke test**

`src/audiobook_creator/__init__.py`:

```python
__version__ = "0.1.0"
```

`tests/test_smoke.py`:

```python
import audiobook_creator


def test_package_imports():
    assert audiobook_creator.__version__ == "0.1.0"
```

`README.md`:

```markdown
# audiobook-creator

Turn PDFs and EPUBs into audiobooks (MP3/M4B) with local-first TTS.

## Setup

    uv sync                 # base (EPUB support)
    uv sync --extra pdf     # + PDF/DOCX/HTML via Docling

Requires ffmpeg on PATH. Run `uv run abc doctor` to check your setup.

Spec: docs/superpowers/specs/2026-08-06-audiobook-creator-design.md
```

- [ ] **Step 3: Create the env and run the test**

Run: `uv sync && uv run pytest -v`
Expected: `test_package_imports PASSED`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src tests README.md uv.lock
git commit -m "chore: scaffold audiobook-creator package with uv, pytest, ruff"
```

---

### Task 2: Data models

**Files:**
- Create: `src/audiobook_creator/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces (used by every later task):
  - `BlockType` (str enum: `heading|paragraph|table|figure|footnote|caption`)
  - `Block(type, text="", level=None, image_path=None)`
  - `DocumentMeta(title="Untitled", author=None, cover_path=None)`
  - `Document(meta, blocks)`
  - `Matter` (str enum: `front_matter|body|back_matter`)
  - `Chapter(index, title, matter=Matter.BODY, blocks)`
  - `Mode` (str enum: `verbatim|rewrite|podcast`)
  - `JobConfig(source, mode=Mode.VERBATIM, tts_backend="kokoro", voice="af_heart", local_only=False, use_llm=True, formats=["m4b"])`
  - `StageStatus` (str enum: `pending|running|done|failed`)
  - `STAGES = ["ingest", "structure", "process", "synthesize", "package"]`
  - `JobState(id, config, stages, errors={}, backends_used=[])`

- [ ] **Step 1: Write the failing test**

`tests/test_models.py`:

```python
from audiobook_creator.models import (
    STAGES,
    Block,
    BlockType,
    Chapter,
    Document,
    DocumentMeta,
    JobConfig,
    JobState,
    Matter,
    Mode,
    StageStatus,
)


def test_document_json_roundtrip():
    doc = Document(
        meta=DocumentMeta(title="T", author="A"),
        blocks=[
            Block(type=BlockType.HEADING, text="Ch 1", level=1),
            Block(type=BlockType.PARAGRAPH, text="Hello."),
        ],
    )
    restored = Document.model_validate_json(doc.model_dump_json())
    assert restored == doc
    assert restored.blocks[0].level == 1


def test_jobconfig_defaults():
    cfg = JobConfig(source="book.epub")
    assert cfg.mode is Mode.VERBATIM
    assert cfg.tts_backend == "kokoro"
    assert cfg.local_only is False
    assert cfg.formats == ["m4b"]


def test_stages_constant_order():
    assert STAGES == ["ingest", "structure", "process", "synthesize", "package"]


def test_jobstate_roundtrip():
    state = JobState(
        id="abc12345",
        config=JobConfig(source="x.epub"),
        stages={name: StageStatus.PENDING for name in STAGES},
    )
    restored = JobState.model_validate_json(state.model_dump_json())
    assert restored.stages["ingest"] is StageStatus.PENDING
    assert restored.errors == {}


def test_chapter_defaults_to_body():
    ch = Chapter(index=0, title="Intro", blocks=[])
    assert ch.matter is Matter.BODY
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError`

- [ ] **Step 3: Write the implementation**

`src/audiobook_creator/models.py`:

```python
from enum import StrEnum

from pydantic import BaseModel


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    FOOTNOTE = "footnote"
    CAPTION = "caption"


class Block(BaseModel):
    type: BlockType
    text: str = ""
    level: int | None = None  # headings only
    image_path: str | None = None  # figures only


class DocumentMeta(BaseModel):
    title: str = "Untitled"
    author: str | None = None
    cover_path: str | None = None


class Document(BaseModel):
    meta: DocumentMeta
    blocks: list[Block]


class Matter(StrEnum):
    FRONT = "front_matter"
    BODY = "body"
    BACK = "back_matter"


class Chapter(BaseModel):
    index: int
    title: str
    matter: Matter = Matter.BODY
    blocks: list[Block]


class Mode(StrEnum):
    VERBATIM = "verbatim"
    REWRITE = "rewrite"
    PODCAST = "podcast"


class JobConfig(BaseModel):
    source: str  # file path or URL
    mode: Mode = Mode.VERBATIM
    tts_backend: str = "kokoro"
    voice: str = "af_heart"
    local_only: bool = False
    use_llm: bool = True
    formats: list[str] = ["m4b"]  # any of: "mp3", "m4b"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


STAGES = ["ingest", "structure", "process", "synthesize", "package"]


class JobState(BaseModel):
    id: str
    config: JobConfig
    stages: dict[str, StageStatus]
    errors: dict[str, str] = {}
    backends_used: list[str] = []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/audiobook_creator/models.py tests/test_models.py
git commit -m "feat: pydantic data models for documents, chapters, and jobs"
```

---

### Task 3: Job and working directory

**Files:**
- Create: `src/audiobook_creator/core/__init__.py` (empty), `src/audiobook_creator/core/job.py`
- Test: `tests/test_job.py`

**Interfaces:**
- Consumes: `JobConfig`, `JobState`, `StageStatus`, `STAGES` from `audiobook_creator.models`.
- Produces:
  - `Job.create(jobs_dir: Path, config: JobConfig) -> Job` — makes `jobs/<id>/` + subdirs, saves `job.json`
  - `Job.load(jobs_dir: Path, job_id: str) -> Job`
  - `Job.list_ids(jobs_dir: Path) -> list[str]`
  - `job.save() -> None`
  - `job.state: JobState`
  - Path properties: `job.dir`, `job.document_path`, `job.chapters_dir`, `job.processed_dir`, `job.audio_dir`, `job.output_dir`, `job.assets_dir`

- [ ] **Step 1: Write the failing test**

`tests/test_job.py`:

```python
from pathlib import Path

from audiobook_creator.core.job import Job
from audiobook_creator.models import JobConfig, StageStatus


def test_create_makes_directories_and_state(tmp_path: Path):
    job = Job.create(tmp_path, JobConfig(source="book.epub"))
    assert job.dir == tmp_path / job.state.id
    for d in (job.chapters_dir, job.processed_dir, job.audio_dir, job.output_dir, job.assets_dir):
        assert d.is_dir()
    assert (job.dir / "job.json").is_file()
    assert all(s is StageStatus.PENDING for s in job.state.stages.values())


def test_load_roundtrip(tmp_path: Path):
    created = Job.create(tmp_path, JobConfig(source="book.epub", voice="am_adam"))
    created.state.stages["ingest"] = StageStatus.DONE
    created.save()

    loaded = Job.load(tmp_path, created.state.id)
    assert loaded.state.config.voice == "am_adam"
    assert loaded.state.stages["ingest"] is StageStatus.DONE


def test_list_ids(tmp_path: Path):
    a = Job.create(tmp_path, JobConfig(source="a.epub"))
    b = Job.create(tmp_path, JobConfig(source="b.epub"))
    assert set(Job.list_ids(tmp_path)) == {a.state.id, b.state.id}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_job.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/audiobook_creator/core/job.py`:

```python
import uuid
from pathlib import Path

from audiobook_creator.models import STAGES, JobConfig, JobState, StageStatus


class Job:
    def __init__(self, jobs_dir: Path, state: JobState):
        self.jobs_dir = jobs_dir
        self.state = state

    # --- paths ---
    @property
    def dir(self) -> Path:
        return self.jobs_dir / self.state.id

    @property
    def document_path(self) -> Path:
        return self.dir / "document.json"

    @property
    def chapters_dir(self) -> Path:
        return self.dir / "chapters"

    @property
    def processed_dir(self) -> Path:
        return self.dir / "processed"

    @property
    def audio_dir(self) -> Path:
        return self.dir / "audio"

    @property
    def output_dir(self) -> Path:
        return self.dir / "output"

    @property
    def assets_dir(self) -> Path:
        return self.dir / "assets"

    # --- lifecycle ---
    @classmethod
    def create(cls, jobs_dir: Path, config: JobConfig) -> "Job":
        state = JobState(
            id=uuid.uuid4().hex[:8],
            config=config,
            stages={name: StageStatus.PENDING for name in STAGES},
        )
        job = cls(jobs_dir, state)
        for d in (job.chapters_dir, job.processed_dir, job.audio_dir, job.output_dir, job.assets_dir):
            d.mkdir(parents=True, exist_ok=True)
        job.save()
        return job

    @classmethod
    def load(cls, jobs_dir: Path, job_id: str) -> "Job":
        raw = (jobs_dir / job_id / "job.json").read_text(encoding="utf-8")
        return cls(jobs_dir, JobState.model_validate_json(raw))

    @classmethod
    def list_ids(cls, jobs_dir: Path) -> list[str]:
        if not jobs_dir.is_dir():
            return []
        return sorted(p.parent.name for p in jobs_dir.glob("*/job.json"))

    def save(self) -> None:
        (self.dir / "job.json").write_text(self.state.model_dump_json(indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_job.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/audiobook_creator/core tests/test_job.py
git commit -m "feat: Job with per-job working directory and persisted state"
```

---

### Task 4: Pipeline engine

**Files:**
- Create: `src/audiobook_creator/core/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `Job`, `STAGES`, `StageStatus`.
- Produces:
  - `run(job: Job, from_stage: str | None = None) -> None` — runs non-done stages in order; `from_stage` resets that stage and all later ones to pending first; failures set `StageStatus.FAILED`, record `errors[stage]`, save, and re-raise.
  - `get_stages() -> dict[str, Callable[[Job], None]]` — lazy imports of each stage's `run_stage` (monkeypatchable in tests).

- [ ] **Step 1: Write the failing test**

`tests/test_engine.py`:

```python
from pathlib import Path

import pytest

from audiobook_creator.core import engine
from audiobook_creator.core.job import Job
from audiobook_creator.models import STAGES, JobConfig, StageStatus


def _fake_stages(calls: list[str], fail_at: str | None = None):
    def make(name):
        def stage(job):
            if name == fail_at:
                raise ValueError(f"boom in {name}")
            calls.append(name)
        return stage

    return {name: make(name) for name in STAGES}


def test_runs_all_stages_in_order(tmp_path: Path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(engine, "get_stages", lambda: _fake_stages(calls))
    job = Job.create(tmp_path, JobConfig(source="x.epub"))
    engine.run(job)
    assert calls == STAGES
    assert all(s is StageStatus.DONE for s in job.state.stages.values())


def test_skips_completed_stages(tmp_path: Path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(engine, "get_stages", lambda: _fake_stages(calls))
    job = Job.create(tmp_path, JobConfig(source="x.epub"))
    job.state.stages["ingest"] = StageStatus.DONE
    job.state.stages["structure"] = StageStatus.DONE
    engine.run(job)
    assert calls == ["process", "synthesize", "package"]


def test_failure_recorded_and_raised(tmp_path: Path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(engine, "get_stages", lambda: _fake_stages(calls, fail_at="process"))
    job = Job.create(tmp_path, JobConfig(source="x.epub"))
    with pytest.raises(ValueError):
        engine.run(job)
    assert job.state.stages["process"] is StageStatus.FAILED
    assert "boom in process" in job.state.errors["process"]
    # persisted to disk too
    reloaded = Job.load(tmp_path, job.state.id)
    assert reloaded.state.stages["process"] is StageStatus.FAILED


def test_from_stage_resets_later_stages(tmp_path: Path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(engine, "get_stages", lambda: _fake_stages(calls))
    job = Job.create(tmp_path, JobConfig(source="x.epub"))
    for name in STAGES:
        job.state.stages[name] = StageStatus.DONE
    engine.run(job, from_stage="synthesize")
    assert calls == ["synthesize", "package"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/audiobook_creator/core/engine.py`:

```python
from collections.abc import Callable

from audiobook_creator.core.job import Job
from audiobook_creator.models import STAGES, StageStatus

StageFunc = Callable[[Job], None]


def get_stages() -> dict[str, StageFunc]:
    """Lazy stage lookup so heavy deps import only when their stage runs."""
    from audiobook_creator.ingest.stage import run_stage as ingest
    from audiobook_creator.package.stage import run_stage as package
    from audiobook_creator.process.stage import run_stage as process
    from audiobook_creator.structure.stage import run_stage as structure
    from audiobook_creator.synthesize.stage import run_stage as synthesize

    return {
        "ingest": ingest,
        "structure": structure,
        "process": process,
        "synthesize": synthesize,
        "package": package,
    }


def run(job: Job, from_stage: str | None = None) -> None:
    if from_stage is not None:
        if from_stage not in STAGES:
            raise ValueError(f"unknown stage {from_stage!r}; expected one of {STAGES}")
        for name in STAGES[STAGES.index(from_stage):]:
            job.state.stages[name] = StageStatus.PENDING
        job.save()

    stages = get_stages()
    for name in STAGES:
        if job.state.stages[name] is StageStatus.DONE:
            continue
        job.state.stages[name] = StageStatus.RUNNING
        job.state.errors.pop(name, None)
        job.save()
        try:
            stages[name](job)
        except Exception as exc:
            job.state.stages[name] = StageStatus.FAILED
            job.state.errors[name] = f"{type(exc).__name__}: {exc}"
            job.save()
            raise
        job.state.stages[name] = StageStatus.DONE
        job.save()
```

Note: `get_stages()` imports stage modules that don't exist yet — that's fine because tests monkeypatch it. The imports resolve as Tasks 5–11 land; the CLI e2e (Task 13) exercises the real wiring.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_engine.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/audiobook_creator/core/engine.py tests/test_engine.py
git commit -m "feat: pipeline engine with resume, from-stage reset, and failure capture"
```

---

### Task 5: Rule-based text normalization

**Files:**
- Create: `src/audiobook_creator/process/__init__.py` (empty), `src/audiobook_creator/process/rules.py`
- Test: `tests/test_rules.py`

**Interfaces:**
- Produces: `normalize(text: str) -> str` — drops citation markers, expands speech-hostile abbreviations/symbols, collapses whitespace. Pure function, no LLM.

- [ ] **Step 1: Write the failing test**

`tests/test_rules.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rules.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/audiobook_creator/process/rules.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rules.py -v`
Expected: 12 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/audiobook_creator/process tests/test_rules.py
git commit -m "feat: rule-based TTS text normalization (citations, abbreviations, symbols)"
```

---

### Task 6: EPUB ingestion

**Files:**
- Create: `src/audiobook_creator/ingest/__init__.py` (empty), `src/audiobook_creator/ingest/epub.py`, `tests/conftest.py`
- Test: `tests/test_epub_ingest.py`

**Interfaces:**
- Consumes: `Document`, `DocumentMeta`, `Block`, `BlockType`.
- Produces:
  - `ingest_epub(path: Path, assets_dir: Path) -> Document` — parses OPF metadata (title/author), walks spine XHTML in order, maps `h1..h6 -> HEADING(level)`, `p -> PARAGRAPH`, `table -> TABLE` (flattened text), extracts cover image to `assets_dir/cover.<ext>` when declared.
  - conftest fixture `make_epub(tmp_path) -> Path` used by later tasks (returns a 2-chapter EPUB titled "Test Book" by "Jane Doe", chapters "Chapter One" / "Chapter Two", plus a "References" file).

- [ ] **Step 1: Write the conftest EPUB builder**

`tests/conftest.py`:

```python
import zipfile
from pathlib import Path

import pytest

_CONTAINER = """<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""

_OPF = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">test-book-001</dc:identifier>
    <dc:title>Test Book</dc:title>
    <dc:creator>Jane Doe</dc:creator>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
    <item id="c3" href="refs.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="c1"/>
    <itemref idref="c2"/>
    <itemref idref="c3"/>
  </spine>
</package>
"""

_NAV = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>Nav</title></head>
<body><nav epub:type="toc"><ol>
<li><a href="ch1.xhtml">Chapter One</a></li>
<li><a href="ch2.xhtml">Chapter Two</a></li>
<li><a href="refs.xhtml">References</a></li>
</ol></nav></body></html>
"""

_CH1 = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Ch 1</title></head>
<body>
<h1>Chapter One</h1>
<p>It was a dark and stormy night.</p>
<p>The rain fell in torrents, at 40% intensity.</p>
</body></html>
"""

_CH2 = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Ch 2</title></head>
<body>
<h1>Chapter Two</h1>
<p>Morning came quietly.</p>
<table><tr><td>Year</td><td>2026</td></tr></table>
</body></html>
"""

_REFS = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Refs</title></head>
<body>
<h1>References</h1>
<p>Doe, J. (2026). A study of storms.</p>
</body></html>
"""


@pytest.fixture
def make_epub(tmp_path: Path):
    def _make() -> Path:
        epub_path = tmp_path / "test-book.epub"
        with zipfile.ZipFile(epub_path, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
            zf.writestr("META-INF/container.xml", _CONTAINER)
            zf.writestr("OEBPS/content.opf", _OPF)
            zf.writestr("OEBPS/nav.xhtml", _NAV)
            zf.writestr("OEBPS/ch1.xhtml", _CH1)
            zf.writestr("OEBPS/ch2.xhtml", _CH2)
            zf.writestr("OEBPS/refs.xhtml", _REFS)
        return epub_path

    return _make
```

- [ ] **Step 2: Write the failing test**

`tests/test_epub_ingest.py`:

```python
from pathlib import Path

from audiobook_creator.ingest.epub import ingest_epub
from audiobook_creator.models import BlockType


def test_metadata_extracted(make_epub, tmp_path: Path):
    doc = ingest_epub(make_epub(), tmp_path / "assets")
    assert doc.meta.title == "Test Book"
    assert doc.meta.author == "Jane Doe"


def test_blocks_in_spine_order(make_epub, tmp_path: Path):
    doc = ingest_epub(make_epub(), tmp_path / "assets")
    headings = [b.text for b in doc.blocks if b.type is BlockType.HEADING]
    assert headings == ["Chapter One", "Chapter Two", "References"]
    assert all(b.level == 1 for b in doc.blocks if b.type is BlockType.HEADING)

    first_heading = next(i for i, b in enumerate(doc.blocks) if b.type is BlockType.HEADING)
    paragraphs_after = [
        b.text for b in doc.blocks[first_heading:] if b.type is BlockType.PARAGRAPH
    ]
    assert paragraphs_after[0] == "It was a dark and stormy night."


def test_table_becomes_table_block(make_epub, tmp_path: Path):
    doc = ingest_epub(make_epub(), tmp_path / "assets")
    tables = [b for b in doc.blocks if b.type is BlockType.TABLE]
    assert len(tables) == 1
    assert "2026" in tables[0].text
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_epub_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Write the implementation**

`src/audiobook_creator/ingest/epub.py`:

```python
import posixpath
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup

from audiobook_creator.models import Block, BlockType, Document, DocumentMeta

_CNT_NS = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
_OPF_NS = {"opf": "http://www.idpf.org/2007/opf", "dc": "http://purl.org/dc/elements/1.1/"}


def ingest_epub(path: Path, assets_dir: Path) -> Document:
    assets_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        opf_path = _opf_path(zf)
        opf_root = ET.fromstring(zf.read(opf_path).decode("utf-8"))
        opf_dir = posixpath.dirname(opf_path)

        meta = _metadata(opf_root)
        meta.cover_path = _extract_cover(zf, opf_root, opf_dir, assets_dir)

        blocks: list[Block] = []
        for href in _spine_hrefs(opf_root):
            xhtml = zf.read(posixpath.join(opf_dir, href) if opf_dir else href).decode("utf-8")
            blocks.extend(_blocks_from_xhtml(xhtml))
    return Document(meta=meta, blocks=blocks)


def _opf_path(zf: zipfile.ZipFile) -> str:
    container = ET.fromstring(zf.read("META-INF/container.xml").decode("utf-8"))
    rootfile = container.find(".//c:rootfile", _CNT_NS)
    if rootfile is None:
        raise ValueError("EPUB has no rootfile in META-INF/container.xml")
    return rootfile.attrib["full-path"]


def _metadata(opf_root: ET.Element) -> DocumentMeta:
    title_el = opf_root.find(".//dc:title", _OPF_NS)
    author_el = opf_root.find(".//dc:creator", _OPF_NS)
    return DocumentMeta(
        title=(title_el.text or "Untitled").strip() if title_el is not None else "Untitled",
        author=author_el.text.strip() if author_el is not None and author_el.text else None,
    )


def _manifest(opf_root: ET.Element) -> dict[str, ET.Element]:
    return {
        item.attrib["id"]: item
        for item in opf_root.findall(".//opf:manifest/opf:item", _OPF_NS)
    }


def _spine_hrefs(opf_root: ET.Element) -> list[str]:
    manifest = _manifest(opf_root)
    hrefs: list[str] = []
    for itemref in opf_root.findall(".//opf:spine/opf:itemref", _OPF_NS):
        item = manifest.get(itemref.attrib["idref"])
        if item is not None and "nav" not in item.attrib.get("properties", ""):
            hrefs.append(item.attrib["href"])
    return hrefs


def _extract_cover(
    zf: zipfile.ZipFile, opf_root: ET.Element, opf_dir: str, assets_dir: Path
) -> str | None:
    for item in _manifest(opf_root).values():
        if "cover-image" in item.attrib.get("properties", ""):
            href = item.attrib["href"]
            src = posixpath.join(opf_dir, href) if opf_dir else href
            dest = assets_dir / f"cover{Path(href).suffix}"
            try:
                dest.write_bytes(zf.read(src))
            except KeyError:
                return None
            return str(dest)
    return None


def _blocks_from_xhtml(xhtml: str) -> list[Block]:
    soup = BeautifulSoup(xhtml, "html.parser")
    body = soup.find("body")
    if body is None:
        return []
    blocks: list[Block] = []
    for el in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "table"]):
        text = " ".join(el.get_text(separator=" ").split())
        if not text:
            continue
        if el.name == "p":
            blocks.append(Block(type=BlockType.PARAGRAPH, text=text))
        elif el.name == "table":
            blocks.append(Block(type=BlockType.TABLE, text=text))
        else:
            blocks.append(Block(type=BlockType.HEADING, text=text, level=int(el.name[1])))
    return blocks
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_epub_ingest.py -v`
Expected: 3 PASSED

- [ ] **Step 6: Commit**

```bash
git add src/audiobook_creator/ingest tests/conftest.py tests/test_epub_ingest.py
git commit -m "feat: native EPUB ingestion (zipfile + BeautifulSoup, no AGPL deps)"
```

---

### Task 7: Docling adapter + ingest dispatcher stage

**Files:**
- Create: `src/audiobook_creator/ingest/docling_adapter.py`, `src/audiobook_creator/ingest/stage.py`
- Test: `tests/test_docling_adapter.py`, extend `tests/test_epub_ingest.py` (dispatcher case)

**Interfaces:**
- Consumes: `Document`, `Block`, `BlockType`, `ingest_epub`, `Job`.
- Produces:
  - `document_from_docling(dl_doc) -> Document` — duck-typed over `dl_doc.iterate_items()` yielding `(item, level)`; maps docling labels `title/section_header → HEADING`, `text/paragraph/list_item → PARAGRAPH`, `table → TABLE`, `picture → FIGURE`, `footnote → FOOTNOTE`, `caption → CAPTION`; **drops** `page_header`, `page_footer`, `page_number` unconditionally (spec requirement).
  - `ingest_with_docling(source: str, assets_dir: Path) -> Document` — lazy-imports docling; raises `RuntimeError("PDF/HTML support requires ... uv sync --extra pdf")` if unavailable.
  - `ingest(source: str, assets_dir: Path) -> Document` — dispatcher: `.epub` → `ingest_epub`, everything else (incl. http/https URLs) → docling.
  - `run_stage(job: Job) -> None` — writes `job.document_path` (`document.json`).

- [ ] **Step 1: Write the failing test**

`tests/test_docling_adapter.py`:

```python
from pathlib import Path
from types import SimpleNamespace

import pytest

from audiobook_creator.ingest.docling_adapter import document_from_docling
from audiobook_creator.ingest.stage import ingest, run_stage
from audiobook_creator.core.job import Job
from audiobook_creator.models import BlockType, Document, JobConfig


def _item(label: str, text: str = "", level: int | None = None):
    ns = SimpleNamespace(label=SimpleNamespace(value=label), text=text)
    if level is not None:
        ns.level = level
    return ns


class FakeDoclingDoc:
    name = "My Paper"

    def iterate_items(self):
        items = [
            _item("title", "My Paper"),
            _item("page_header", "Journal of Storms 2026"),
            _item("section_header", "Introduction", level=1),
            _item("text", "Storms are loud."),
            _item("footnote", "1. See appendix."),
            _item("table", "Year 2026 Rain 400mm"),
            _item("picture"),
            _item("caption", "Figure 1: A storm."),
            _item("page_footer", "Page 3"),
        ]
        return [(i, 0) for i in items]


def test_labels_mapped_and_furniture_dropped():
    doc = document_from_docling(FakeDoclingDoc())
    types = [b.type for b in doc.blocks]
    assert BlockType.HEADING in types
    assert BlockType.TABLE in types
    assert BlockType.FIGURE in types
    assert BlockType.CAPTION in types
    assert BlockType.FOOTNOTE in types
    texts = " ".join(b.text for b in doc.blocks)
    assert "Journal of Storms" not in texts  # page_header dropped
    assert "Page 3" not in texts  # page_footer dropped
    assert doc.meta.title == "My Paper"


def test_heading_levels_preserved():
    doc = document_from_docling(FakeDoclingDoc())
    intro = next(b for b in doc.blocks if b.text == "Introduction")
    assert intro.type is BlockType.HEADING
    assert intro.level == 1


def test_dispatcher_routes_epub(make_epub, tmp_path: Path):
    doc = ingest(str(make_epub()), tmp_path / "assets")
    assert isinstance(doc, Document)
    assert doc.meta.title == "Test Book"


def test_dispatcher_pdf_without_docling_gives_actionable_error(tmp_path: Path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_docling(name, *args, **kwargs):
        if name.startswith("docling"):
            raise ImportError("No module named 'docling'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_docling)
    with pytest.raises(RuntimeError, match="--extra pdf"):
        ingest("paper.pdf", tmp_path / "assets")


def test_run_stage_writes_document_json(make_epub, tmp_path: Path):
    job = Job.create(tmp_path / "jobs", JobConfig(source=str(make_epub())))
    run_stage(job)
    assert job.document_path.is_file()
    doc = Document.model_validate_json(job.document_path.read_text(encoding="utf-8"))
    assert doc.meta.title == "Test Book"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_docling_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/audiobook_creator/ingest/docling_adapter.py`:

```python
from pathlib import Path

from audiobook_creator.models import Block, BlockType, Document, DocumentMeta

# docling label value -> our BlockType; None = drop (page furniture)
_LABEL_MAP: dict[str, BlockType | None] = {
    "title": BlockType.HEADING,
    "section_header": BlockType.HEADING,
    "text": BlockType.PARAGRAPH,
    "paragraph": BlockType.PARAGRAPH,
    "list_item": BlockType.PARAGRAPH,
    "table": BlockType.TABLE,
    "picture": BlockType.FIGURE,
    "footnote": BlockType.FOOTNOTE,
    "caption": BlockType.CAPTION,
    "page_header": None,
    "page_footer": None,
    "page_number": None,
}


def _label_value(item) -> str:
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label))


def document_from_docling(dl_doc) -> Document:
    blocks: list[Block] = []
    title: str | None = None
    for item, _level in dl_doc.iterate_items():
        label = _label_value(item)
        if label not in _LABEL_MAP:
            continue
        block_type = _LABEL_MAP[label]
        if block_type is None:
            continue
        text = " ".join((getattr(item, "text", "") or "").split())
        if label == "title" and title is None:
            title = text
        if block_type is BlockType.HEADING:
            level = getattr(item, "level", 1) or 1
            if text:
                blocks.append(Block(type=block_type, text=text, level=int(level)))
        elif block_type is BlockType.FIGURE:
            blocks.append(Block(type=block_type, text=text))
        elif text:
            blocks.append(Block(type=block_type, text=text))
    meta = DocumentMeta(title=title or getattr(dl_doc, "name", None) or "Untitled")
    return Document(meta=meta, blocks=blocks)


def ingest_with_docling(source: str, assets_dir: Path) -> Document:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise RuntimeError(
            "PDF/DOCX/HTML ingestion requires Docling. Install it with: uv sync --extra pdf"
        ) from exc
    result = DocumentConverter().convert(source)
    return document_from_docling(result.document)
```

`src/audiobook_creator/ingest/stage.py`:

```python
from pathlib import Path

from audiobook_creator.core.job import Job
from audiobook_creator.ingest.docling_adapter import ingest_with_docling
from audiobook_creator.ingest.epub import ingest_epub
from audiobook_creator.models import Document


def ingest(source: str, assets_dir: Path) -> Document:
    if not source.startswith(("http://", "https://")) and source.lower().endswith(".epub"):
        return ingest_epub(Path(source), assets_dir)
    return ingest_with_docling(source, assets_dir)


def run_stage(job: Job) -> None:
    doc = ingest(job.state.config.source, job.assets_dir)
    if not doc.blocks:
        raise ValueError(f"ingestion produced no content from {job.state.config.source!r}")
    job.document_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_docling_adapter.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/audiobook_creator/ingest tests/test_docling_adapter.py
git commit -m "feat: docling adapter with page-furniture dropping and ingest dispatcher stage"
```

---

### Task 8: Structure stage (chapters + matter classification)

**Files:**
- Create: `src/audiobook_creator/structure/__init__.py` (empty), `src/audiobook_creator/structure/chapters.py`, `src/audiobook_creator/structure/stage.py`
- Test: `tests/test_structure.py`

**Interfaces:**
- Consumes: `Document`, `Chapter`, `Matter`, `Block`, `BlockType`, `Job`.
- Produces:
  - `split_chapters(doc: Document) -> list[Chapter]` — splits at level-1 headings; if that yields <2 chapters and there are ≥2 level-2 headings, splits at level ≤2 instead; blocks before the first split heading become chapter "Beginning".
  - `classify_matter(title: str) -> Matter` — keyword heuristics; ambiguous → `Matter.BODY` (spec: worst case is extra audio, never missing content).
  - `run_stage(job: Job) -> None` — reads `document.json`, writes `chapters/NNN.json` (3-digit, zero-padded, one `Chapter` per file).

- [ ] **Step 1: Write the failing test**

`tests/test_structure.py`:

```python
from pathlib import Path

from audiobook_creator.core.job import Job
from audiobook_creator.models import (
    Block,
    BlockType,
    Chapter,
    Document,
    DocumentMeta,
    JobConfig,
    Matter,
)
from audiobook_creator.structure.chapters import classify_matter, split_chapters
from audiobook_creator.structure.stage import run_stage


def _doc(blocks: list[Block]) -> Document:
    return Document(meta=DocumentMeta(title="T"), blocks=blocks)


def _h(text: str, level: int = 1) -> Block:
    return Block(type=BlockType.HEADING, text=text, level=level)


def _p(text: str) -> Block:
    return Block(type=BlockType.PARAGRAPH, text=text)


def test_split_on_level1_headings():
    doc = _doc([_h("One"), _p("a"), _h("Two"), _p("b")])
    chapters = split_chapters(doc)
    assert [c.title for c in chapters] == ["One", "Two"]
    assert chapters[0].index == 0
    assert chapters[1].blocks[0].text == "b"


def test_leading_blocks_become_beginning_chapter():
    doc = _doc([_p("preamble"), _h("One"), _p("a")])
    chapters = split_chapters(doc)
    assert chapters[0].title == "Beginning"
    assert chapters[0].blocks[0].text == "preamble"


def test_fallback_to_level2_when_single_level1():
    doc = _doc([_h("Paper Title"), _h("Intro", 2), _p("a"), _h("Methods", 2), _p("b")])
    chapters = split_chapters(doc)
    titles = [c.title for c in chapters]
    assert "Intro" in titles and "Methods" in titles


def test_classify_matter_keywords():
    assert classify_matter("References") is Matter.BACK
    assert classify_matter("Bibliography") is Matter.BACK
    assert classify_matter("Index") is Matter.BACK
    assert classify_matter("Table of Contents") is Matter.FRONT
    assert classify_matter("Copyright") is Matter.FRONT
    assert classify_matter("Preface") is Matter.FRONT
    assert classify_matter("Chapter One") is Matter.BODY
    assert classify_matter("Some Odd Title") is Matter.BODY  # ambiguous -> body


def test_run_stage_writes_chapter_files(tmp_path: Path):
    job = Job.create(tmp_path, JobConfig(source="x.epub"))
    doc = _doc([_h("One"), _p("a"), _h("References"), _p("Doe 2026")])
    job.document_path.write_text(doc.model_dump_json(), encoding="utf-8")
    run_stage(job)
    files = sorted(job.chapters_dir.glob("*.json"))
    assert [f.name for f in files] == ["000.json", "001.json"]
    ch1 = Chapter.model_validate_json(files[1].read_text(encoding="utf-8"))
    assert ch1.matter is Matter.BACK
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_structure.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/audiobook_creator/structure/chapters.py`:

```python
from audiobook_creator.models import Block, BlockType, Chapter, Document, Matter

_FRONT_KEYWORDS = {
    "contents", "table of contents", "title page", "copyright", "dedication",
    "preface", "foreword", "epigraph", "half title", "about this book",
}
_BACK_KEYWORDS = {
    "references", "bibliography", "index", "acknowledgments", "acknowledgements",
    "appendix", "notes", "glossary", "about the author", "endnotes",
}


def classify_matter(title: str) -> Matter:
    t = title.strip().lower()
    if any(t == k or t.startswith(k) for k in _BACK_KEYWORDS):
        return Matter.BACK
    if any(t == k or t.startswith(k) for k in _FRONT_KEYWORDS):
        return Matter.FRONT
    return Matter.BODY  # ambiguous -> body: extra audio beats missing content


def _split_at_level(doc: Document, max_level: int) -> list[Chapter]:
    chapters: list[Chapter] = []
    current_title = "Beginning"
    current_blocks: list[Block] = []

    def flush():
        if current_blocks or (chapters and current_title != "Beginning"):
            chapters.append(
                Chapter(
                    index=len(chapters),
                    title=current_title,
                    matter=classify_matter(current_title),
                    blocks=list(current_blocks),
                )
            )

    for block in doc.blocks:
        if block.type is BlockType.HEADING and (block.level or 1) <= max_level:
            flush()
            current_title = block.text
            current_blocks = []
        else:
            current_blocks.append(block)
    flush()
    return chapters


def split_chapters(doc: Document) -> list[Chapter]:
    chapters = _split_at_level(doc, max_level=1)
    if len(chapters) < 2:
        level2_count = sum(
            1 for b in doc.blocks if b.type is BlockType.HEADING and b.level == 2
        )
        if level2_count >= 2:
            chapters = _split_at_level(doc, max_level=2)
    return chapters
```

`src/audiobook_creator/structure/stage.py`:

```python
from audiobook_creator.core.job import Job
from audiobook_creator.models import Document
from audiobook_creator.structure.chapters import split_chapters


def run_stage(job: Job) -> None:
    doc = Document.model_validate_json(job.document_path.read_text(encoding="utf-8"))
    chapters = split_chapters(doc)
    if not chapters:
        raise ValueError("structure stage produced no chapters")
    job.chapters_dir.mkdir(parents=True, exist_ok=True)
    for chapter in chapters:
        path = job.chapters_dir / f"{chapter.index:03d}.json"
        path.write_text(chapter.model_dump_json(indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_structure.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/audiobook_creator/structure tests/test_structure.py
git commit -m "feat: chapter splitting with level fallback and front/back-matter classification"
```

---

### Task 9: Verbatim processing stage

**Files:**
- Create: `src/audiobook_creator/process/verbatim.py`, `src/audiobook_creator/process/stage.py`
- Test: `tests/test_verbatim.py`

**Interfaces:**
- Consumes: `Chapter`, `BlockType`, `Matter`, `normalize`, `Job`.
- Produces:
  - `render_chapter_text(chapter: Chapter) -> str` — headings become `"<text>. [[pause]]"`; paragraphs/captions normalized; TABLE/FIGURE/FOOTNOTE blocks skipped (v1 verbatim contract); paragraphs joined with blank lines.
  - `run_stage(job: Job) -> None` — for each `chapters/NNN.json` with `matter == BODY`, writes `processed/NNN.txt`. Raises if zero body chapters.
  - The `[[pause]]` marker (exact string) — consumed by synthesize (Task 11).

- [ ] **Step 1: Write the failing test**

`tests/test_verbatim.py`:

```python
from pathlib import Path

from audiobook_creator.core.job import Job
from audiobook_creator.models import (
    Block,
    BlockType,
    Chapter,
    JobConfig,
    Matter,
)
from audiobook_creator.process.verbatim import render_chapter_text
from audiobook_creator.process.stage import run_stage


def test_render_heading_pause_and_normalized_paragraphs():
    ch = Chapter(
        index=0,
        title="One",
        blocks=[
            Block(type=BlockType.HEADING, text="Chapter One", level=1),
            Block(type=BlockType.PARAGRAPH, text="Rain fell at 40% intensity [3]."),
            Block(type=BlockType.TABLE, text="Year 2026"),
            Block(type=BlockType.FOOTNOTE, text="1. ignore me"),
            Block(type=BlockType.CAPTION, text="Figure 1: A storm."),
        ],
    )
    text = render_chapter_text(ch)
    assert text.startswith("Chapter One. [[pause]]")
    assert "40 percent intensity." in text
    assert "Year 2026" not in text  # tables skipped in verbatim v1
    assert "ignore me" not in text  # footnotes skipped
    assert "Figure 1: A storm." in text  # captions kept


def test_run_stage_writes_body_chapters_only(tmp_path: Path):
    job = Job.create(tmp_path, JobConfig(source="x.epub"))
    body = Chapter(
        index=0, title="One",
        blocks=[Block(type=BlockType.PARAGRAPH, text="Hello there.")],
    )
    back = Chapter(
        index=1, title="References", matter=Matter.BACK,
        blocks=[Block(type=BlockType.PARAGRAPH, text="Doe 2026.")],
    )
    (job.chapters_dir / "000.json").write_text(body.model_dump_json(), encoding="utf-8")
    (job.chapters_dir / "001.json").write_text(back.model_dump_json(), encoding="utf-8")
    run_stage(job)
    files = sorted(job.processed_dir.glob("*.txt"))
    assert [f.name for f in files] == ["000.txt"]
    assert "Hello there." in files[0].read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_verbatim.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**

`src/audiobook_creator/process/verbatim.py`:

```python
from audiobook_creator.models import BlockType, Chapter
from audiobook_creator.process.rules import normalize

PAUSE = "[[pause]]"

# Verbatim v1 contract: read prose and captions; skip tables, figures, and
# footnote bodies (their inline markers are stripped by normalize()).
_SKIPPED = {BlockType.TABLE, BlockType.FIGURE, BlockType.FOOTNOTE}


def render_chapter_text(chapter: Chapter) -> str:
    parts: list[str] = []
    for block in chapter.blocks:
        if block.type in _SKIPPED:
            continue
        text = normalize(block.text)
        if not text:
            continue
        if block.type is BlockType.HEADING:
            parts.append(f"{text.rstrip('.')}. {PAUSE}")
        else:
            parts.append(text)
    return "\n\n".join(parts)
```

`src/audiobook_creator/process/stage.py`:

```python
from audiobook_creator.core.job import Job
from audiobook_creator.models import Chapter, Matter, Mode
from audiobook_creator.process.verbatim import render_chapter_text


def run_stage(job: Job) -> None:
    mode = job.state.config.mode
    if mode is not Mode.VERBATIM:
        raise NotImplementedError(
            f"mode {mode.value!r} lands in Plan 2 (LLM layer); only 'verbatim' works today"
        )
    job.processed_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in sorted(job.chapters_dir.glob("*.json")):
        chapter = Chapter.model_validate_json(path.read_text(encoding="utf-8"))
        if chapter.matter is not Matter.BODY:
            continue
        text = render_chapter_text(chapter)
        if not text.strip():
            continue
        (job.processed_dir / f"{chapter.index:03d}.txt").write_text(text, encoding="utf-8")
        written += 1
    if written == 0:
        raise ValueError("no body chapters produced speakable text")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_verbatim.py -v`
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/audiobook_creator/process tests/test_verbatim.py
git commit -m "feat: verbatim processing stage (rule-based, body chapters only)"
```

---

### Task 10: TTS foundation — chunking, backend registry, stub backend

**Files:**
- Create: `src/audiobook_creator/synthesize/__init__.py` (empty), `src/audiobook_creator/synthesize/base.py`, `src/audiobook_creator/synthesize/stub.py`
- Test: `tests/test_tts_base.py`

**Interfaces:**
- Consumes: nothing project-internal.
- Produces:
  - `chunk_text(text: str, max_chars: int = 400) -> list[str]` — sentence-aware, never splits mid-sentence unless a single sentence exceeds `max_chars` (then hard-splits on spaces).
  - `class TTSBackend(Protocol)`: attributes `name: str`, `sample_rate: int`; method `synthesize(text: str, voice: str) -> bytes` returning raw mono 16-bit little-endian PCM.
  - `class PrivacyError(RuntimeError)`.
  - `register_backend(name: str, factory: Callable[[], TTSBackend], is_local: bool)` and `get_backend(name: str, local_only: bool = False) -> TTSBackend` — raises `PrivacyError` for non-local backends when `local_only=True`; raises `ValueError` for unknown names.
  - `write_wav(path: Path, pcm: bytes, sample_rate: int) -> None` and `wav_duration_seconds(path: Path) -> float` (stdlib `wave`).
  - `StubBackend` — `name="stub"`, `sample_rate=24000`, 440 Hz tone, 10 ms of audio per character; registered as local. Kokoro registers in Task 14.

- [ ] **Step 1: Write the failing test**

`tests/test_tts_base.py`:

```python
from pathlib import Path

import pytest

from audiobook_creator.synthesize.base import (
    PrivacyError,
    chunk_text,
    get_backend,
    register_backend,
    wav_duration_seconds,
    write_wav,
)
from audiobook_creator.synthesize.stub import StubBackend


def test_chunk_respects_sentences():
    text = "First sentence here. Second one is also short. Third."
    chunks = chunk_text(text, max_chars=45)
    assert all(len(c) <= 45 for c in chunks)
    assert all(c.endswith((".", "!", "?")) for c in chunks)  # sentence boundaries only
    assert " ".join(chunks) == text


def test_chunk_hard_splits_monster_sentence():
    text = "word " * 200  # one 1000-char "sentence"
    chunks = chunk_text(text.strip(), max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) >= 9


def test_chunk_empty_returns_empty():
    assert chunk_text("   ") == []


def test_stub_backend_duration_scales_with_text():
    stub = StubBackend()
    short = stub.synthesize("hi", "any")
    long = stub.synthesize("hello there friend", "any")
    assert len(long) > len(short)
    assert len(short) % 2 == 0  # 16-bit samples


def test_registry_returns_stub():
    backend = get_backend("stub")
    assert backend.name == "stub"


def test_registry_unknown_name():
    with pytest.raises(ValueError, match="unknown TTS backend"):
        get_backend("nope")


def test_privacy_blocks_network_backends():
    register_backend("fake-cloud", StubBackend, is_local=False)
    with pytest.raises(PrivacyError):
        get_backend("fake-cloud", local_only=True)
    assert get_backend("fake-cloud", local_only=False).name == "stub"


def test_wav_roundtrip(tmp_path: Path):
    pcm = b"\x00\x01" * 24000  # exactly 1 second at 24 kHz mono 16-bit
    path = tmp_path / "t.wav"
    write_wav(path, pcm, 24000)
    assert abs(wav_duration_seconds(path) - 1.0) < 0.001
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tts_base.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/audiobook_creator/synthesize/base.py`:

```python
import re
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class PrivacyError(RuntimeError):
    """Raised when a job flagged local_only requests a network backend."""


class TTSBackend(Protocol):
    name: str
    sample_rate: int

    def synthesize(self, text: str, voice: str) -> bytes:
        """Return raw mono 16-bit little-endian PCM at self.sample_rate."""
        ...


_REGISTRY: dict[str, tuple[Callable[[], TTSBackend], bool]] = {}


def register_backend(name: str, factory: Callable[[], TTSBackend], is_local: bool) -> None:
    _REGISTRY[name] = (factory, is_local)


def get_backend(name: str, local_only: bool = False) -> TTSBackend:
    if name not in _REGISTRY:
        raise ValueError(f"unknown TTS backend {name!r}; known: {sorted(_REGISTRY)}")
    factory, is_local = _REGISTRY[name]
    if local_only and not is_local:
        raise PrivacyError(
            f"backend {name!r} sends text to a network service, but this job is local_only"
        )
    return factory()


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    out: list[str] = []
    while len(sentence) > max_chars:
        cut = sentence.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        out.append(sentence[:cut].strip())
        sentence = sentence[cut:].strip()
    if sentence:
        out.append(sentence)
    return out


def chunk_text(text: str, max_chars: int = 400) -> list[str]:
    text = text.strip()
    if not text:
        return []
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        if sentence:
            pieces.extend(_hard_split(sentence, max_chars))
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= max_chars:
            current = f"{current} {piece}"
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()
```

`src/audiobook_creator/synthesize/stub.py`:

```python
import numpy as np

from audiobook_creator.synthesize.base import register_backend


class StubBackend:
    """Deterministic tone generator: 10 ms of 440 Hz per character. For tests/CI."""

    name = "stub"
    sample_rate = 24000

    def synthesize(self, text: str, voice: str) -> bytes:
        n_samples = max(1, int(0.010 * len(text) * self.sample_rate))
        t = np.arange(n_samples, dtype=np.float64)
        samples = 0.2 * np.sin(2 * np.pi * 440.0 * t / self.sample_rate)
        return (samples * 32767).astype("<i2").tobytes()


register_backend("stub", StubBackend, is_local=True)
```

Registration on import: `base.get_backend` must see registered backends. Add to `src/audiobook_creator/synthesize/__init__.py`:

```python
from audiobook_creator.synthesize import stub  # noqa: F401  (registers "stub")
```

And in `tests/test_tts_base.py` the import of `StubBackend` triggers registration.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tts_base.py -v`
Expected: 8 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/audiobook_creator/synthesize tests/test_tts_base.py
git commit -m "feat: TTS foundation - chunking, backend registry with privacy gate, stub backend"
```

---

### Task 11: Synthesize stage (cache, retry, per-chapter WAVs)

**Files:**
- Create: `src/audiobook_creator/synthesize/stage.py`
- Test: `tests/test_synthesize_stage.py`

**Interfaces:**
- Consumes: `Job`, `get_backend`, `chunk_text`, `write_wav`, `wav_duration_seconds`, `PAUSE` marker convention from Task 9.
- Produces:
  - `run_stage(job: Job) -> None` — for each `processed/NNN.txt`, writes `audio/NNN.wav` (skips if it exists). Splits on `[[pause]]`, inserts 0.7 s silence between segments. Per-chunk PCM cached at `audio/cache/<sha1>.pcm` keyed by `backend|voice|chunk`. Chunk synthesis failure: 2 retries, then 0.3 s silence + `logging` warning (job never dies on one chunk). Appends `"tts:<backend>"` to `job.state.backends_used` (deduped).

- [ ] **Step 1: Write the failing test**

`tests/test_synthesize_stage.py`:

```python
from pathlib import Path

from audiobook_creator.core.job import Job
from audiobook_creator.models import JobConfig
from audiobook_creator.synthesize import stage as synth_stage
from audiobook_creator.synthesize.base import register_backend, wav_duration_seconds


class FlakyBackend:
    """Fails permanently for chunks containing 'BAD'."""

    name = "flaky"
    sample_rate = 24000

    def synthesize(self, text: str, voice: str) -> bytes:
        if "BAD" in text:
            raise RuntimeError("synthesis exploded")
        return b"\x01\x00" * int(0.010 * len(text) * self.sample_rate)


register_backend("flaky", FlakyBackend, is_local=True)


def _job_with_processed(tmp_path: Path, texts: dict[str, str], backend: str = "stub") -> Job:
    job = Job.create(tmp_path, JobConfig(source="x.epub", tts_backend=backend))
    for name, text in texts.items():
        (job.processed_dir / name).write_text(text, encoding="utf-8")
    return job


def test_produces_wav_per_chapter(tmp_path: Path):
    job = _job_with_processed(
        tmp_path, {"000.txt": "Hello world. [[pause]] Next section.", "001.txt": "Short."}
    )
    synth_stage.run_stage(job)
    wavs = sorted(job.audio_dir.glob("*.wav"))
    assert [w.name for w in wavs] == ["000.wav", "001.wav"]
    # pause adds 0.7s silence: chapter 0 must be longer than its text alone implies
    assert wav_duration_seconds(wavs[0]) > 0.7
    assert "tts:stub" in job.state.backends_used


def test_chunk_cache_is_populated_and_reused(tmp_path: Path):
    job = _job_with_processed(tmp_path, {"000.txt": "Hello world."})
    synth_stage.run_stage(job)
    cache_files = list((job.audio_dir / "cache").glob("*.pcm"))
    assert len(cache_files) >= 1
    mtime = cache_files[0].stat().st_mtime_ns

    (job.audio_dir / "000.wav").unlink()  # force resynthesis
    synth_stage.run_stage(job)
    assert cache_files[0].stat().st_mtime_ns == mtime  # reused, not rewritten


def test_existing_wavs_skipped(tmp_path: Path):
    job = _job_with_processed(tmp_path, {"000.txt": "Hello."})
    synth_stage.run_stage(job)
    wav = job.audio_dir / "000.wav"
    mtime = wav.stat().st_mtime_ns
    synth_stage.run_stage(job)
    assert wav.stat().st_mtime_ns == mtime


def test_failed_chunk_becomes_silence_not_crash(tmp_path: Path, caplog):
    job = _job_with_processed(tmp_path, {"000.txt": "Good text. BAD text. More good."},
                              backend="flaky")
    synth_stage.run_stage(job)  # must not raise
    assert (job.audio_dir / "000.wav").exists()
    assert any("BAD" in r.message or "failed" in r.message.lower() for r in caplog.records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_synthesize_stage.py -v`
Expected: FAIL with `ImportError` (no `stage` module)

- [ ] **Step 3: Write the implementation**

`src/audiobook_creator/synthesize/stage.py`:

```python
import hashlib
import logging

from audiobook_creator.core.job import Job
from audiobook_creator.synthesize.base import TTSBackend, chunk_text, get_backend, write_wav

logger = logging.getLogger(__name__)

PAUSE = "[[pause]]"
_PAUSE_SECONDS = 0.7
_FAIL_SILENCE_SECONDS = 0.3
_RETRIES = 2


def _silence(seconds: float, sample_rate: int) -> bytes:
    return b"\x00\x00" * int(seconds * sample_rate)


def _synth_chunk(backend: TTSBackend, text: str, voice: str) -> bytes:
    last_exc: Exception | None = None
    for _attempt in range(_RETRIES + 1):
        try:
            return backend.synthesize(text, voice)
        except Exception as exc:  # noqa: BLE001 - backend errors are non-fatal by spec
            last_exc = exc
    logger.warning(
        "TTS failed after %d attempts, inserting silence for chunk %r: %s",
        _RETRIES + 1, text[:60], last_exc,
    )
    return _silence(_FAIL_SILENCE_SECONDS, backend.sample_rate)


def run_stage(job: Job) -> None:
    cfg = job.state.config
    backend = get_backend(cfg.tts_backend, local_only=cfg.local_only)
    used = f"tts:{backend.name}"
    if used not in job.state.backends_used:
        job.state.backends_used.append(used)
        job.save()

    cache_dir = job.audio_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for txt_path in sorted(job.processed_dir.glob("*.txt")):
        wav_path = job.audio_dir / f"{txt_path.stem}.wav"
        if wav_path.exists():
            continue
        pcm = bytearray()
        segments = txt_path.read_text(encoding="utf-8").split(PAUSE)
        for i, segment in enumerate(segments):
            for chunk in chunk_text(segment):
                key = hashlib.sha1(
                    f"{backend.name}|{cfg.voice}|{chunk}".encode("utf-8")
                ).hexdigest()
                cached = cache_dir / f"{key}.pcm"
                if cached.exists():
                    data = cached.read_bytes()
                else:
                    data = _synth_chunk(backend, chunk, cfg.voice)
                    cached.write_bytes(data)
                pcm += data
            if i < len(segments) - 1:
                pcm += _silence(_PAUSE_SECONDS, backend.sample_rate)
        if not pcm:
            logger.warning("chapter %s produced no audio, skipping", txt_path.stem)
            continue
        write_wav(wav_path, bytes(pcm), backend.sample_rate)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_synthesize_stage.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/audiobook_creator/synthesize/stage.py tests/test_synthesize_stage.py
git commit -m "feat: synthesize stage with chunk cache, retry-to-silence, and pause handling"
```

---

### Task 12: Package stage (MP3s + M4B via ffmpeg)

**Files:**
- Create: `src/audiobook_creator/package/__init__.py` (empty), `src/audiobook_creator/package/ffmpeg.py`, `src/audiobook_creator/package/stage.py`
- Create: `tests/helpers.py` (ffmpeg skip marker)
- Test: `tests/test_package_stage.py`

**Interfaces:**
- Consumes: `Job`, `Chapter`, `wav_duration_seconds`, `DocumentMeta` (from `document.json` for title/author/cover).
- Produces:
  - `ffmpeg_available() -> bool` (both ffmpeg and ffprobe on PATH).
  - `encode_mp3(wav: Path, mp3: Path, *, title: str, artist: str | None, album: str, track: int) -> None`
  - `write_ffmetadata(path: Path, *, title: str, artist: str | None, chapters: list[tuple[str, float]]) -> None` — chapters as (title, duration_seconds), ms timebase, cumulative offsets.
  - `build_m4b(wavs: list[Path], meta_path: Path, out: Path, cover: Path | None) -> None`
  - `probe_chapters(m4b: Path) -> list[str]` — chapter titles via ffprobe JSON (used by tests).
  - `run_stage(job: Job) -> None` — reads formats from config; writes `output/NNN - <safe title>.mp3` per chapter and/or `output/<safe book title>.m4b`.
  - `safe_filename(name: str) -> str` — strips `<>:"/\|?*` and trims.

- [ ] **Step 1: Add ffmpeg skip helper**

Create `tests/helpers.py` (imported as `from helpers import ...` — pytest puts `tests/` on `sys.path`; `tests.conftest` is NOT importable without an `__init__.py`):

```python
import shutil

import pytest

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)
```

- [ ] **Step 2: Write the failing test**

`tests/test_package_stage.py`:

```python
import json
import subprocess
from pathlib import Path

from audiobook_creator.core.job import Job
from audiobook_creator.models import (
    Block,
    BlockType,
    Chapter,
    Document,
    DocumentMeta,
    JobConfig,
)
from audiobook_creator.package.ffmpeg import safe_filename, write_ffmetadata
from audiobook_creator.package.stage import run_stage
from audiobook_creator.synthesize.base import write_wav
from helpers import requires_ffmpeg


def test_safe_filename():
    assert safe_filename('Ch: "One" <draft?>') == "Ch One draft"


def test_ffmetadata_chapter_offsets(tmp_path: Path):
    meta = tmp_path / "meta.txt"
    write_ffmetadata(meta, title="Book", artist="Jane",
                     chapters=[("One", 2.0), ("Two", 3.5)])
    content = meta.read_text(encoding="utf-8")
    assert content.startswith(";FFMETADATA1")
    assert "title=Book" in content and "artist=Jane" in content
    assert "START=0" in content and "END=2000" in content
    assert "START=2000" in content and "END=5500" in content
    assert "title=One" in content and "title=Two" in content


def _prepared_job(tmp_path: Path, formats: list[str]) -> Job:
    job = Job.create(tmp_path, JobConfig(source="x.epub", formats=formats))
    doc = Document(meta=DocumentMeta(title="Test Book", author="Jane Doe"),
                   blocks=[Block(type=BlockType.PARAGRAPH, text="x")])
    job.document_path.write_text(doc.model_dump_json(), encoding="utf-8")
    for i, title in enumerate(["One", "Two"]):
        ch = Chapter(index=i, title=title, blocks=[])
        (job.chapters_dir / f"{i:03d}.json").write_text(ch.model_dump_json(), encoding="utf-8")
        write_wav(job.audio_dir / f"{i:03d}.wav", b"\x00\x01" * 24000, 24000)  # 1s each
    return job


@requires_ffmpeg
def test_m4b_with_chapters(tmp_path: Path):
    job = _prepared_job(tmp_path, ["m4b"])
    run_stage(job)
    m4b = job.output_dir / "Test Book.m4b"
    assert m4b.exists()
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_chapters", str(m4b)],
        check=True, capture_output=True, text=True,
    )
    chapters = json.loads(probe.stdout)["chapters"]
    assert [c["tags"]["title"] for c in chapters] == ["One", "Two"]


@requires_ffmpeg
def test_mp3_per_chapter(tmp_path: Path):
    job = _prepared_job(tmp_path, ["mp3"])
    run_stage(job)
    mp3s = sorted(job.output_dir.glob("*.mp3"))
    assert [m.name for m in mp3s] == ["000 - One.mp3", "001 - Two.mp3"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_package_stage.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Write the implementation**

`src/audiobook_creator/package/ffmpeg.py`:

```python
import json
import re
import shutil
import subprocess
from pathlib import Path


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def safe_filename(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "", name)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _run(args: list[str]) -> None:
    try:
        subprocess.run(args, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"ffmpeg failed ({' '.join(args[:3])}...): {stderr}") from exc


def encode_mp3(wav: Path, mp3: Path, *, title: str, artist: str | None, album: str,
               track: int) -> None:
    args = ["ffmpeg", "-y", "-i", str(wav), "-c:a", "libmp3lame", "-q:a", "4",
            "-metadata", f"title={title}", "-metadata", f"album={album}",
            "-metadata", f"track={track}"]
    if artist:
        args += ["-metadata", f"artist={artist}"]
    args.append(str(mp3))
    _run(args)


def _escape_meta(value: str) -> str:
    # FFMETADATA escapes: '=', ';', '#', '\' and newline
    return re.sub(r"([=;#\\\n])", r"\\\1", value)


def write_ffmetadata(path: Path, *, title: str, artist: str | None,
                     chapters: list[tuple[str, float]]) -> None:
    lines = [";FFMETADATA1", f"title={_escape_meta(title)}"]
    if artist:
        lines.append(f"artist={_escape_meta(artist)}")
    offset_ms = 0
    for chapter_title, duration_s in chapters:
        end_ms = offset_ms + int(round(duration_s * 1000))
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={offset_ms}", f"END={end_ms}",
                  f"title={_escape_meta(chapter_title)}"]
        offset_ms = end_ms
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_m4b(wavs: list[Path], meta_path: Path, out: Path, cover: Path | None) -> None:
    list_path = out.parent / "concat.txt"
    escaped = [w.as_posix().replace("'", "'\\''") for w in wavs]
    list_path.write_text("".join(f"file '{p}'\n" for p in escaped), encoding="utf-8")
    args = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-i", str(meta_path)]
    if cover is not None and cover.exists():
        args += ["-i", str(cover), "-map", "0:a", "-map", "2:v", "-c:v", "mjpeg",
                 "-disposition:v:0", "attached_pic"]
    else:
        args += ["-map", "0:a"]
    args += ["-map_metadata", "1", "-c:a", "aac", "-b:a", "64k", "-f", "mp4", str(out)]
    _run(args)
    list_path.unlink(missing_ok=True)


def probe_chapters(m4b: Path) -> list[str]:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_chapters", str(m4b)],
        check=True, capture_output=True, text=True,
    )
    return [c["tags"]["title"] for c in json.loads(result.stdout)["chapters"]]
```

`src/audiobook_creator/package/stage.py`:

```python
from pathlib import Path

from audiobook_creator.core.job import Job
from audiobook_creator.models import Chapter, Document
from audiobook_creator.package.ffmpeg import (
    build_m4b,
    encode_mp3,
    ffmpeg_available,
    safe_filename,
    write_ffmetadata,
)
from audiobook_creator.synthesize.base import wav_duration_seconds


def run_stage(job: Job) -> None:
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg/ffprobe not found on PATH. Install ffmpeg and retry; "
                           "see 'abc doctor'.")
    doc = Document.model_validate_json(job.document_path.read_text(encoding="utf-8"))
    titles = {
        (c := Chapter.model_validate_json(p.read_text(encoding="utf-8"))).index: c.title
        for p in job.chapters_dir.glob("*.json")
    }
    wavs = sorted(job.audio_dir.glob("*.wav"))
    if not wavs:
        raise ValueError("no chapter audio found; did synthesize run?")

    chapters: list[tuple[str, float]] = []
    for wav in wavs:
        index = int(wav.stem)
        chapters.append((titles.get(index, f"Chapter {index + 1}"), wav_duration_seconds(wav)))

    formats = job.state.config.formats
    if "mp3" in formats:
        for wav, (title, _dur) in zip(wavs, chapters):
            mp3 = job.output_dir / f"{wav.stem} - {safe_filename(title)}.mp3"
            encode_mp3(wav, mp3, title=title, artist=doc.meta.author,
                       album=doc.meta.title, track=int(wav.stem) + 1)
    if "m4b" in formats:
        meta_path = job.output_dir / "ffmetadata.txt"
        write_ffmetadata(meta_path, title=doc.meta.title, artist=doc.meta.author,
                         chapters=chapters)
        cover = Path(doc.meta.cover_path) if doc.meta.cover_path else None
        out = job.output_dir / f"{safe_filename(doc.meta.title)}.m4b"
        build_m4b(wavs, meta_path, out, cover)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_package_stage.py -v`
Expected: 4 PASSED (or 2 PASSED + 2 SKIPPED without ffmpeg — on the dev machine ffmpeg must be installed, so expect 4)

- [ ] **Step 6: Commit**

```bash
git add src/audiobook_creator/package tests/test_package_stage.py tests/helpers.py
git commit -m "feat: package stage - per-chapter MP3s and M4B with chapter markers via ffmpeg"
```

---

### Task 13: CLI + end-to-end test

**Files:**
- Create: `src/audiobook_creator/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `Job`, `engine.run`, `JobConfig`, `Mode`, `STAGES`, `ffmpeg_available`, `probe_chapters`.
- Produces: Typer app `app` with commands:
  - `abc convert SOURCE [--mode verbatim] [--tts-backend kokoro] [--voice af_heart] [--format m4b]... [--local-only] [--jobs-dir jobs] [--from-stage STAGE]`
  - `abc jobs [--jobs-dir jobs]` — table: id, source, mode, per-stage status summary
  - `abc resume JOB_ID [--jobs-dir jobs] [--from-stage STAGE]`
  - `abc doctor` — checks ffmpeg/ffprobe, docling importability, kokoro model files; exit 1 if ffmpeg missing

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:

```python
from pathlib import Path

from typer.testing import CliRunner

from audiobook_creator.cli import app
from helpers import requires_ffmpeg

runner = CliRunner()


@requires_ffmpeg
def test_end_to_end_epub_to_m4b(make_epub, tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    result = runner.invoke(app, [
        "convert", str(make_epub()),
        "--tts-backend", "stub", "--format", "m4b", "--jobs-dir", str(jobs_dir),
    ])
    assert result.exit_code == 0, result.output
    m4bs = list(jobs_dir.glob("*/output/*.m4b"))
    assert len(m4bs) == 1
    from audiobook_creator.package.ffmpeg import probe_chapters
    # fixture EPUB: References chapter is back matter and excluded -> 2 chapters
    assert probe_chapters(m4bs[0]) == ["Chapter One", "Chapter Two"]


def test_jobs_lists_created_job(make_epub, tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    from audiobook_creator.core.job import Job
    from audiobook_creator.models import JobConfig

    job = Job.create(jobs_dir, JobConfig(source="whatever.epub"))
    result = runner.invoke(app, ["jobs", "--jobs-dir", str(jobs_dir)])
    assert result.exit_code == 0
    assert job.state.id in result.output


def test_convert_rejects_missing_file(tmp_path: Path):
    result = runner.invoke(app, ["convert", str(tmp_path / "nope.epub"),
                                 "--jobs-dir", str(tmp_path / "jobs")])
    assert result.exit_code != 0


def test_doctor_runs():
    result = runner.invoke(app, ["doctor"])
    assert "ffmpeg" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/audiobook_creator/cli.py`:

```python
import shutil
from pathlib import Path

import typer

from audiobook_creator.core import engine
from audiobook_creator.core.job import Job
from audiobook_creator.models import JobConfig, Mode, StageStatus

app = typer.Typer(help="Turn PDFs and EPUBs into audiobooks.", no_args_is_help=True)

_STATUS_ICON = {
    StageStatus.PENDING: "·",
    StageStatus.RUNNING: "~",
    StageStatus.DONE: "+",
    StageStatus.FAILED: "x",
}


@app.command()
def convert(
    source: str = typer.Argument(..., help="Path to EPUB/PDF/DOCX or URL"),
    mode: Mode = typer.Option(Mode.VERBATIM, "--mode"),
    tts_backend: str = typer.Option("kokoro", "--tts-backend"),
    voice: str = typer.Option("af_heart", "--voice"),
    formats: list[str] = typer.Option(["m4b"], "--format", help="m4b and/or mp3"),
    local_only: bool = typer.Option(False, "--local-only",
                                    help="Hard-block all network backends for this job"),
    jobs_dir: Path = typer.Option(Path("jobs"), "--jobs-dir"),
) -> None:
    if not source.startswith(("http://", "https://")) and not Path(source).is_file():
        typer.echo(f"error: source file not found: {source}", err=True)
        raise typer.Exit(code=2)
    for fmt in formats:
        if fmt not in ("mp3", "m4b"):
            typer.echo(f"error: unknown format {fmt!r} (use mp3 or m4b)", err=True)
            raise typer.Exit(code=2)
    config = JobConfig(source=source, mode=mode, tts_backend=tts_backend, voice=voice,
                       local_only=local_only, formats=formats)
    job = Job.create(jobs_dir, config)
    typer.echo(f"job {job.state.id}: {source}")
    engine.run(job)
    for out in sorted(job.output_dir.iterdir()):
        if out.suffix in (".m4b", ".mp3"):
            typer.echo(f"  -> {out}")


@app.command()
def resume(
    job_id: str,
    jobs_dir: Path = typer.Option(Path("jobs"), "--jobs-dir"),
    from_stage: str = typer.Option(None, "--from-stage",
                                   help="Reset this stage and later ones, then run"),
) -> None:
    job = Job.load(jobs_dir, job_id)
    engine.run(job, from_stage=from_stage)
    typer.echo(f"job {job_id} complete")


@app.command()
def jobs(jobs_dir: Path = typer.Option(Path("jobs"), "--jobs-dir")) -> None:
    ids = Job.list_ids(jobs_dir)
    if not ids:
        typer.echo("no jobs")
        return
    for job_id in ids:
        job = Job.load(jobs_dir, job_id)
        stages = " ".join(
            f"{name}:{_STATUS_ICON[status]}" for name, status in job.state.stages.items()
        )
        typer.echo(f"{job_id}  {job.state.config.mode.value:<8}  {stages}  "
                   f"{job.state.config.source}")


@app.command()
def doctor() -> None:
    problems = 0

    ffmpeg = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
    typer.echo(f"ffmpeg/ffprobe: {'OK' if ffmpeg else 'MISSING - install ffmpeg, add to PATH'}")
    if not ffmpeg:
        problems += 1

    try:
        import docling  # noqa: F401
        typer.echo("docling (PDF support): OK")
    except ImportError:
        typer.echo("docling (PDF support): not installed - EPUB only. "
                   "Install: uv sync --extra pdf")

    from audiobook_creator.synthesize.kokoro import model_files_status
    status = model_files_status()
    typer.echo(f"kokoro models: {status}")

    raise typer.Exit(code=1 if problems else 0)


if __name__ == "__main__":
    app()
```

Note: `doctor` imports `model_files_status` from Task 14's module. To keep this task self-contained and green, Task 13 creates a minimal placeholder `src/audiobook_creator/synthesize/kokoro.py`:

```python
import os
from pathlib import Path

MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"


def models_dir() -> Path:
    return Path(os.environ.get("ABC_MODELS_DIR", "models"))


def model_files_status() -> str:
    missing = [f for f in (MODEL_FILE, VOICES_FILE) if not (models_dir() / f).is_file()]
    if not missing:
        return "OK"
    return (f"missing {', '.join(missing)} in {models_dir()}/ - download from "
            "https://github.com/thewh1teagle/kokoro-onnx/releases (or set ABC_MODELS_DIR)")
```

Task 14 extends this file with the actual backend class.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v && uv run pytest -v`
Expected: CLI tests pass; full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/audiobook_creator/cli.py src/audiobook_creator/synthesize/kokoro.py tests/test_cli.py
git commit -m "feat: Typer CLI (convert/resume/jobs/doctor) with stub-backend e2e test"
```

---

### Task 14: Kokoro backend + preview command

**Files:**
- Modify: `src/audiobook_creator/synthesize/kokoro.py` (add backend class + registration), `src/audiobook_creator/synthesize/__init__.py` (import kokoro module), `src/audiobook_creator/cli.py` (add `preview` command)
- Test: `tests/test_kokoro.py`, extend `tests/test_cli.py`

**Interfaces:**
- Consumes: `register_backend`, `write_wav`, `chunk_text`, `Job`, `get_backend`.
- Produces:
  - `KokoroBackend` — `name="kokoro"`, `sample_rate=24000`; loads `kokoro-v1.0.onnx` + `voices-v1.0.bin` from `models_dir()`; raises `RuntimeError` with download instructions when files are missing; converts kokoro float32 output to 16-bit PCM. Registered as local.
  - `abc preview JOB_ID` — synthesizes the first 300 chars of the first processed chapter to `output/preview.wav` using the job's configured backend/voice.

- [ ] **Step 1: Write the failing tests**

`tests/test_kokoro.py`:

```python
from pathlib import Path

import pytest

from audiobook_creator.synthesize.kokoro import KokoroBackend, models_dir


def test_missing_models_give_actionable_error(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ABC_MODELS_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="kokoro-onnx/releases"):
        KokoroBackend()


def test_registered_as_local():
    from audiobook_creator.synthesize.base import _REGISTRY
    assert "kokoro" in _REGISTRY
    assert _REGISTRY["kokoro"][1] is True  # is_local


@pytest.mark.kokoro
def test_real_synthesis_produces_audio(monkeypatch):
    # Requires real model files under ./models (or ABC_MODELS_DIR). Run with: -m kokoro
    backend = KokoroBackend()
    pcm = backend.synthesize("Hello world.", "af_heart")
    assert len(pcm) > 24000  # > 0.5s of 16-bit audio
```

Append to `tests/test_cli.py`:

```python
def test_preview_requires_processed_text(tmp_path: Path):
    from audiobook_creator.core.job import Job
    from audiobook_creator.models import JobConfig

    jobs_dir = tmp_path / "jobs"
    job = Job.create(jobs_dir, JobConfig(source="x.epub", tts_backend="stub"))
    result = runner.invoke(app, ["preview", job.state.id, "--jobs-dir", str(jobs_dir)])
    assert result.exit_code != 0  # no processed text yet

    (job.processed_dir / "000.txt").write_text("Hello preview world.", encoding="utf-8")
    result = runner.invoke(app, ["preview", job.state.id, "--jobs-dir", str(jobs_dir)])
    assert result.exit_code == 0, result.output
    assert (job.output_dir / "preview.wav").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_kokoro.py tests/test_cli.py -v`
Expected: FAIL (`KokoroBackend` not defined; `preview` command missing)

- [ ] **Step 3: Write the implementation**

Replace `src/audiobook_creator/synthesize/kokoro.py` with:

```python
import os
from pathlib import Path

import numpy as np

from audiobook_creator.synthesize.base import register_backend

MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"


def models_dir() -> Path:
    return Path(os.environ.get("ABC_MODELS_DIR", "models"))


def model_files_status() -> str:
    missing = [f for f in (MODEL_FILE, VOICES_FILE) if not (models_dir() / f).is_file()]
    if not missing:
        return "OK"
    return (f"missing {', '.join(missing)} in {models_dir()}/ - download from "
            "https://github.com/thewh1teagle/kokoro-onnx/releases (or set ABC_MODELS_DIR)")


class KokoroBackend:
    name = "kokoro"
    sample_rate = 24000

    def __init__(self) -> None:
        onnx_path = models_dir() / MODEL_FILE
        voices_path = models_dir() / VOICES_FILE
        if not onnx_path.is_file() or not voices_path.is_file():
            raise RuntimeError(model_files_status())
        from kokoro_onnx import Kokoro

        self._kokoro = Kokoro(str(onnx_path), str(voices_path))

    def synthesize(self, text: str, voice: str) -> bytes:
        samples, sample_rate = self._kokoro.create(text, voice=voice, speed=1.0)
        if sample_rate != self.sample_rate:
            raise RuntimeError(f"unexpected kokoro sample rate {sample_rate}")
        clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
        return (clipped * 32767).astype("<i2").tobytes()


register_backend("kokoro", KokoroBackend, is_local=True)
```

Update `src/audiobook_creator/synthesize/__init__.py`:

```python
from audiobook_creator.synthesize import kokoro, stub  # noqa: F401  (register backends)
```

Add to `src/audiobook_creator/cli.py` (new command; add `PREVIEW_CHARS = 300` near the top constants):

```python
@app.command()
def preview(
    job_id: str,
    jobs_dir: Path = typer.Option(Path("jobs"), "--jobs-dir"),
) -> None:
    """Synthesize ~30s from the first processed chapter to output/preview.wav."""
    from audiobook_creator.synthesize.base import chunk_text, get_backend, write_wav

    job = Job.load(jobs_dir, job_id)
    processed = sorted(job.processed_dir.glob("*.txt"))
    if not processed:
        typer.echo("error: no processed text yet - run convert (or resume) first", err=True)
        raise typer.Exit(code=2)
    text = processed[0].read_text(encoding="utf-8").replace("[[pause]]", " ")[:300]
    cfg = job.state.config
    backend = get_backend(cfg.tts_backend, local_only=cfg.local_only)
    pcm = b"".join(backend.synthesize(c, cfg.voice) for c in chunk_text(text))
    out = job.output_dir / "preview.wav"
    write_wav(out, pcm, backend.sample_rate)
    typer.echo(f"  -> {out}")
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -v`
Expected: all PASS (kokoro real-synthesis test auto-skipped via marker)

- [ ] **Step 5: Manual smoke test on real hardware (documented, not automated)**

Download models, then:

```bash
mkdir -p models
# download kokoro-v1.0.onnx and voices-v1.0.bin from
# https://github.com/thewh1teagle/kokoro-onnx/releases into models/
uv run abc doctor
uv run abc convert path/to/some-real-book.epub --format m4b
uv run pytest -m kokoro -v   # real synthesis test
```

Listen to the first minute of the output. This step verifies voice quality and Windows/AMD execution; note observations in the commit message body.

- [ ] **Step 6: Commit**

```bash
git add src/audiobook_creator/synthesize src/audiobook_creator/cli.py tests/test_kokoro.py tests/test_cli.py
git commit -m "feat: kokoro-onnx local TTS backend and preview command"
```

---

## Plan Self-Review (completed at write time)

- **Spec coverage (Plan 1 scope):** ingest with furniture-dropping ✔ (Task 7); structure + matter classification with no-LLM fallback ✔ (Task 8); verbatim with rule-based path ✔ (Tasks 5, 9); sentence-aware chunking, per-chunk cache, retry→silence ✔ (Tasks 10, 11); MP3 + M4B with chapters/metadata/cover ✔ (Task 12); CLI convert/resume/jobs/doctor/preview ✔ (Tasks 13, 14); resumable file-contract jobs ✔ (Tasks 3, 4); privacy gate on backend selection ✔ (Task 10). Deferred by design: LLM clients/modes (Plan 2), web UI (Plan 3), figure image extraction to assets (Plan 2, rewrite mode needs it).
- **Placeholders:** none — every step carries runnable code or exact commands.
- **Type consistency:** `run_stage(job)` signature uniform across all five stage modules; `chunk_text`/`write_wav`/`get_backend` signatures match between Tasks 10, 11, 13, 14; `PAUSE` constant defined in `process/verbatim.py` and mirrored as a string in `synthesize/stage.py` (kept as separate constants intentionally — synthesize must not import process).
