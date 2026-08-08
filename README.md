# audiobook-creator

Turn EPUBs and PDFs into audiobooks — per-chapter MP3s or an M4B with chapter
markers — using local-first TTS. Runs entirely on your machine.

## Prerequisites

- **Python 3.13** managed with [uv](https://docs.astral.sh/uv/)
- **ffmpeg + ffprobe on PATH** — e.g. `winget install Gyan.FFmpeg` (restart your
  shell afterwards so the PATH change is picked up by `uv run`)
- **Kokoro model files** (local TTS voice, ~340 MB total) in `models/`:

      mkdir models
      curl -L -o models/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
      curl -L -o models/voices-v1.0.bin  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

  Set `ABC_MODELS_DIR` to keep them elsewhere.

## Setup

    uv sync                 # base install (EPUB support)
    uv sync --extra pdf     # + PDF/DOCX/HTML via Docling (pulls the PyTorch stack)
    uv run abc doctor       # verify ffmpeg, docling, and model files

## Usage

    uv run abc convert book.epub                          # -> M4B with chapters
    uv run abc convert paper.pdf --format mp3             # per-chapter MP3s
    uv run abc convert book.epub --format m4b --format mp3
    uv run abc convert report.pdf --local-only            # hard-block network backends
    uv run abc convert book.epub --voice am_adam          # pick a Kokoro voice

    uv run abc jobs                                       # list jobs + stage status
    uv run abc resume <job-id>                            # continue a failed/interrupted job
    uv run abc preview <job-id>                           # ~30s audition -> output/preview.wav
    uv run abc doctor                                     # environment checkup

Outputs land in `jobs/<job-id>/output/`.

## The job directory

Every conversion is a resumable job with inspectable intermediate files:

    jobs/<id>/
    ├── job.json          # config + per-stage status + warnings
    ├── document.json     # extracted document (ingest)
    ├── chapters/NNN.json # chapter split + front/body/back classification (structure)
    ├── processed/NNN.txt # speakable text, human-editable (process)
    ├── audio/NNN.wav     # per-chapter audio + chunk cache (synthesize)
    └── output/           # final MP3s / M4B (package)

Each stage skips work that already exists, so `abc resume` continues where a
crash or Ctrl-C left off — long TTS runs never start over.

## Fix the text, keep the audio

The processed text is meant to be edited. If the narration mangles something:

1. Edit `jobs/<id>/processed/NNN.txt` (plain text; `[[pause]]` marks pauses)
2. `uv run abc resume <id> --from-stage synthesize`

Only chapters whose text changed are re-synthesized; unchanged chunks come from
the cache.

**Include or exclude chapters:** the structure stage classifies chapters as
`front_matter` / `body` / `back_matter`, and only `body` is narrated. If a
chapter was misclassified, edit its `"matter"` field in
`jobs/<id>/chapters/NNN.json` and run
`uv run abc resume <id> --from-stage process`.

## Project status

v0.1 (Plan 1 of 3): verbatim narration with rule-based cleanup, Kokoro local
TTS, EPUB + PDF ingest, MP3/M4B packaging. Coming next: LLM-powered modes
(spoken-friendly rewrite, podcast digest, figure/table verbalization) and a web
UI. Design docs live in `docs/superpowers/specs/`.
