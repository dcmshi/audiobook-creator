# Audiobook Creator — Design Spec

**Date:** 2026-08-06
**Status:** Approved (brainstorm complete)

## Purpose

Convert PDFs, EPUBs, and web articles into listenable audio (MP3s and M4B audiobooks) for on-the-go consumption. Existing tools (audiblez, abogen, ebook2audiobook) handle clean fiction EPUBs well but uniformly mishandle technical PDFs — headers, footers, citations, and table debris get read aloud. The differentiated core of this project is a document-understanding + LLM processing layer that makes technical material listenable, with prose EPUBs handled as the easy case.

## Requirements

- **Inputs:** EPUB, PDF (including technical papers/textbooks), DOCX, HTML/URLs.
- **Modes (per job):**
  - `verbatim` — faithful reading, light normalization only.
  - `rewrite` — same content re-expressed for the ear (tables/figures/citations verbalized).
  - `podcast` — condensed two-host conversational digest.
- **Outputs:** per-chapter MP3s and/or single M4B with chapter markers, metadata, cover art.
- **TTS:** local-first (Kokoro via ONNX), API backends (OpenAI, Gemini) as quality escape hatch.
- **LLM:** frontier API by default; per-job `local_only` privacy flag routes to Ollama or disables LLM steps. Work documents must never leave the machine.
- **Interfaces:** CLI and minimal web UI, both in v1, over one shared engine.
- **Environment:** Windows 11, AMD RX 7900 XT (20 GB, no CUDA) — base install must not require PyTorch/CUDA.

## Architecture

Single Python package with a five-stage pipeline. The contract between stages is files on disk in a per-job directory; each stage is idempotent and skipped if its output exists (this is the resume mechanism).

```
audiobook_creator/
├── core/           # pipeline engine: Job, stage runner, resume logic
├── ingest/         # Docling wrapper → normalized document.json
├── structure/      # chapter splitting, front/back-matter classification
├── process/        # LLM layer: verbatim | rewrite | podcast
├── synthesize/     # TTS backends: kokoro (default), openai, gemini, stub
├── package/        # ffmpeg → MP3s / M4B with chapters
├── cli.py          # Typer CLI ("abc")
└── web/            # FastAPI + single-page frontend (htmx/vanilla, SSE progress)
```

**Job working directory:**

```
jobs/<id>/
├── job.json          # source, config, per-stage status, errors, backend audit log
├── document.json     # ingest output
├── chapters/NN.json  # structure output
├── processed/NN.txt  # process output (human-editable plain text)
├── audio/            # per-chunk cache + per-chapter WAVs
└── output/           # final MP3s / M4B
```

### Stack

| Concern | Choice | Rationale |
|---|---|---|
| Language/env | Python 3.12+, `uv` | ML ecosystem; fast reproducible envs |
| Extraction | Docling (MIT) | One path for PDF+EPUB+DOCX+HTML; labels headers/footers/footnotes/tables/figures; avoids AGPL (`ebooklib`, `pymupdf4llm`) |
| Local TTS | `kokoro-onnx` | 82M params, CPU-viable, DirectML-capable on AMD; no PyTorch in base install |
| API TTS | OpenAI, Gemini TTS | ~$9–22/book; pluggable backend interface |
| LLM | Thin client interface: Anthropic/Gemini API impl + Ollama impl | Per-job privacy enforcement at the factory |
| Audio | ffmpeg (external binary) | MP3 encode, M4B mux with FFMETADATA chapters |
| CLI | Typer | |
| Web | FastAPI + htmx/vanilla JS, SSE | No frontend build chain; localhost, single-user, no auth in v1 |

Heavier local TTS (Chatterbox, Orpheus via WSL2+ROCm) is explicitly out of scope for v1; the backend interface leaves room for it.

## Stage specifications

### 1. Ingest — `source → document.json`

Docling converts the source into an ordered list of typed blocks: `heading[level]`, `paragraph`, `table`, `figure` (image extracted to file, caption attached), `footnote`, `caption`, `page_header`, `page_footer`. Page headers/footers and bare page numbers are dropped unconditionally — no mode wants them. URLs are fetched and ingested via the HTML path. Cover image extracted when present.

### 2. Structure — `document.json → chapters/NN.json`

Split into chapters via heading hierarchy (EPUB TOC when available). Each chapter is classified `front_matter` / `body` / `back_matter` — heuristics first (position, title keywords like "References", "Index", "Copyright"), cheap LLM call as tiebreaker when ambiguous. Non-body chapters are kept but flagged and default-excluded from audio; CLI/web expose include/exclude overrides.

### 3. Process — `chapters/*.json → processed/NN.txt`

Output is plain speakable text; the only markup is `[[pause]]` at section boundaries.

- **verbatim:** chunk-by-chunk LLM normalization with a strict change-nothing-else prompt: expand numbers/abbreviations/units, drop citation markers ("[37]", superscripts) and footnote references, "Fig. 3" → "Figure 3". A rule-based fallback (`--no-llm`, or per-chunk on validation failure) keeps the pipeline functional with no LLM configured.
- **rewrite:** content-preserving re-expression for listening. Tables → spoken summary; figures → VLM description of the extracted image, spliced at the reference point; equations → spoken or intuitive description; citation clusters → natural attribution ("Smith and colleagues found…").
- **podcast:** whole-document pass producing a condensed two-speaker dialogue script (speaker-tagged lines); rendered with two voices in synthesis.

Validation: verbatim chunks are length-ratio checked (catches the LLM summarizing when it must not); failures fall back to rule-based normalization for that chunk and are logged.

### 4. Synthesize — `processed/*.txt → audio/NN.wav`

Sentence-aware chunking (~400 chars, never splitting mid-sentence) → TTS backend → per-chapter concatenation. Backends implement one interface: `synthesize(text, voice) → wav bytes`; registered: `kokoro` (default), `openai`, `gemini`, `stub` (tone generator, for tests). Podcast speaker tags map to distinct voices. Per-chunk WAVs cached on disk keyed by (text-hash, backend, voice) — a crash at chapter 18/20 resumes at full speed.

### 5. Package — `audio/*.wav → output/`

ffmpeg produces per-chapter MP3s (ID3: title, author, track numbers) and/or one M4B (AAC) with FFMETADATA chapter markers, metadata, and cover art. Both formats can be requested in one job.

## Job model and frontends

A job = source + config (`mode`, `tts_backend`, `voice`, `privacy`, `output_formats`, chapter include/exclude) with per-stage status in `job.json`. The engine runs stages in order, skipping completed ones. `--from-stage` forces re-runs. Processed text is deliberately human-editable between stages.

**CLI (reference frontend):**

```
abc convert paper.pdf --mode rewrite
abc convert novel.epub --mode verbatim --format m4b
abc convert report.pdf --local-only
abc jobs | abc resume <id> | abc preview <id>   # preview: first ~30s of chapter 1
```

**Web UI:** upload → mode/voice/privacy pickers → live per-stage progress (SSE) → play/download; textarea editing of processed text with re-run-from-synthesize. Same engine and `jobs/` dir as the CLI. Localhost only in v1.

## Privacy

`privacy: local_only` is enforced at client construction: requesting any network backend (LLM or TTS) raises before a request is made. `job.json` records every backend that touched content, as an audit trail. Default for all jobs is API-allowed; the flag is per-job.

## Error handling

- A stage failure marks the job `failed` at that stage, error recorded in `job.json`; `resume` retries from the failed stage.
- TTS chunk failure: 2 retries, then insert silence, log warning with chunk list. Long jobs never die at 95%.
- LLM validation failure (verbatim): per-chunk fallback to rules, logged.
- Missing ffmpeg / missing model weights: detected at startup with actionable messages (`abc doctor` command).

## Testing

- Per-stage unit tests on small fixtures: a 3-page PDF (table + figure + footnotes + running headers), a 2-chapter EPUB; snapshot tests for normalization rules.
- LLM-dependent tests run against recorded responses; `--live` marker for real-API smoke tests.
- End-to-end: tiny EPUB → M4B via `stub` TTS backend; assert chapter markers via ffprobe. CI needs no weights, GPU, or API keys.

## Out of scope for v1

- RSS/podcast feed hosting, Audiobookshelf integration
- Voice cloning; PyTorch-based TTS backends (Chatterbox/Orpheus)
- Multi-user web deployment, auth
- Read-along/subtitle sync output
- Non-English content

## Open questions deferred to implementation planning

- Exact LLM prompt designs per mode (iterate against fixtures)
- Kokoro voice default and DirectML vs CPU execution-provider selection
- Whether Docling's HTML fetch suffices for paywalled/JS-heavy articles (fallback: accept saved HTML/PDF)
