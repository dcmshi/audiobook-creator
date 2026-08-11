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
    # Podcast mode only: [[speaker:N]] picks podcast_voices[N-1], falling back to `voice`.
    podcast_voices: list[str] = ["af_heart", "am_adam"]
    local_only: bool = False
    use_llm: bool = True
    # None = local-first default; "anthropic"/"kimi"/"ollama" force one
    llm_provider: str | None = None
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
