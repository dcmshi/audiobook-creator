# audiobook-creator

Turn PDFs and EPUBs into audiobooks (MP3/M4B) with local-first TTS.

## Setup

    uv sync                 # base (EPUB support)
    uv sync --extra pdf     # + PDF/DOCX/HTML via Docling

Requires ffmpeg on PATH. Run `uv run abc doctor` to check your setup.

Spec: docs/superpowers/specs/2026-08-06-audiobook-creator-design.md
