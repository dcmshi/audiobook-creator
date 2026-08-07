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
