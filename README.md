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

## LLM-powered modes

Three text modes. `verbatim` needs no LLM; the other two do.

| Mode | What it produces | LLM |
|---|---|---|
| `verbatim` (default) | The book read as written, with abbreviations, numbers and citation markers cleaned up | Optional — improves the cleanup, falls back to rules |
| `rewrite` | The same content re-expressed for listening: tables summarised aloud, figures described, citations spoken naturally | Required |
| `podcast` | A two-host conversation covering the whole document, rendered in two voices | Required |

    uv run abc convert book.epub --mode rewrite
    uv run abc convert paper.pdf --mode podcast --llm anthropic
    uv run abc convert book.epub --no-llm                  # rule-based text only

### Providers

| Provider | Cost | How it is chosen |
|---|---|---|
| `ollama` | Free, runs on your machine | **Default.** Used automatically when Ollama is running |
| `anthropic` | Paid API | Only with `--llm anthropic` or `ABC_LLM=anthropic` |
| `kimi` | Paid API | Only with `--llm kimi` or `ABC_LLM=kimi` |

A paid provider never runs unless you ask for it. With no flag and no Ollama
running, `verbatim` quietly uses its rule-based path and the other two modes
refuse to start rather than spending money you did not authorise.

> **Billing warning — a chat subscription is not API access.** Claude Pro/Max
> and Kimi memberships do **not** include API usage. Both vendors bill the API
> separately, per token, against an API key you create yourself. This is exactly
> why paid providers are opt-in per job.

Rough cost for a full book: **`claude-opus-5` ≈ $5–15**; **`kimi-k2.6` ≈ 5x
cheaper**; **`claude-haiku-4-5`** is Anthropic's cheap tier
(`ABC_LLM_MODEL=claude-haiku-4-5`). `ollama` is free but slow on a CPU — fine
for `verbatim` and `rewrite`, which are chunked, and painful for `podcast`,
which asks for one long script in a single call.

### Environment variables

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic credentials |
| `ABC_LLM_MODEL` | Anthropic model (default `claude-opus-5`) |
| `MOONSHOT_API_KEY` | Kimi credentials |
| `ABC_KIMI_MODEL` | Kimi model (default `kimi-k2.6`) |
| `ABC_KIMI_URL` | Kimi endpoint (default `https://api.moonshot.ai/v1`) |
| `ABC_OLLAMA_MODEL` | Ollama model (default `qwen3:14b`) |
| `ABC_OLLAMA_URL` | Ollama endpoint (default `http://localhost:11434`) |
| `ABC_OLLAMA_NUM_CTX` | Ollama context window (default `8192`) |
| `ABC_LLM` | Force a provider without the flag: `anthropic`, `kimi`, `ollama`, or `none` |

`uv run abc doctor` reports which of the three are usable.

### Privacy

`--local-only` hard-blocks every network backend for that job, so work
documents stay on the machine. Under `--local-only` the Anthropic and Kimi
providers are refused outright, and Ollama is accepted only when its URL points
at this machine — a LAN or hosted Ollama is still off-box and is rejected
before a single request is sent.

`--no-llm` skips the LLM entirely and uses the rule-based text path. It applies
to `verbatim` only; the other two modes have nothing to fall back to and will
tell you so.

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

v0.2 (Plans 1-2 of 3): verbatim narration with rule-based cleanup, Kokoro
local TTS, EPUB + PDF ingest, MP3/M4B packaging, plus the LLM layer —
spoken-friendly `rewrite`, two-voice `podcast`, figure and table verbalization,
and figure descriptions via vision. Coming next: a web UI. Design docs live in
`docs/superpowers/specs/`.
