"""Reject an unusable job before any work starts, for whichever frontend asked.

Split out of the CLI so the web app cannot drift from it: a job the CLI refuses and the web
accepts would fail three stages later, with a job directory already on disk.
"""

from pathlib import Path

from audiobook_creator.models import JobConfig, Mode

_FORMATS = ("mp3", "m4b")
_PROVIDERS = ("anthropic", "kimi", "ollama")


class PreflightError(ValueError):
    """A job that cannot succeed. The message is written to be shown to a person."""


def preflight(config: JobConfig, *, check_source: bool = True) -> list[str]:
    """Raise PreflightError on anything fatal; return the non-fatal warnings to display.

    `check_source` is off for uploads, where the bytes arrive with the request and the
    config's source path does not exist yet.
    """
    # Imported here rather than at module scope: loading the TTS backend registry and the
    # LLM package costs real time, and `abc jobs` should not pay it.
    from audiobook_creator.process import llm as llm_pkg
    from audiobook_creator.synthesize.base import check_backend
    from audiobook_creator.synthesize.kokoro import model_files_status

    warnings: list[str] = []

    if (
        check_source
        and not config.source.startswith(("http://", "https://"))
        and not Path(config.source).is_file()
    ):
        raise PreflightError(f"source file not found: {config.source}")

    for fmt in config.formats:
        if fmt not in _FORMATS:
            raise PreflightError(f"unknown format {fmt!r} (use mp3 or m4b)")

    if config.llm_provider is not None and config.llm_provider not in _PROVIDERS:
        raise PreflightError(
            f"unknown LLM provider {config.llm_provider!r} (use anthropic, kimi, or ollama)"
        )
    if config.llm_provider and not config.use_llm:
        raise PreflightError("--llm and --no-llm cannot be used together")

    if config.mode is not Mode.VERBATIM:
        if not config.use_llm:
            raise PreflightError(
                f"mode {config.mode.value!r} needs an LLM; --no-llm only works with "
                "--mode verbatim"
            )
        # Constructed once and discarded: every client's constructor is a cheap probe, so
        # this costs a socket check rather than a stage run.
        try:
            client = llm_pkg.resolve_llm(
                local_only=config.local_only, use_llm=True, provider=config.llm_provider
            )
        except Exception as exc:  # noqa: BLE001 - PrivacyError and friends are user-facing
            raise PreflightError(str(exc)) from None
        if client is None:
            raise PreflightError(
                f"mode {config.mode.value!r} needs an LLM: start Ollama, pass "
                "--llm anthropic / --llm kimi (paid APIs, billed per token), "
                "or use --mode verbatim"
            )
        if config.mode is Mode.PODCAST and getattr(client, "name", "") == "ollama":
            # Not a refusal: it works, it is just a poor fit. Podcast sends the whole book in
            # one prompt and asks for one long script, which is where local models struggle.
            warnings.append(
                "podcast mode on a local model covers only as much of the document as fits "
                "its context window, and a full script can take about an hour on CPU. "
                "Consider --llm anthropic or --llm kimi."
            )

    try:
        check_backend(config.tts_backend, local_only=config.local_only)
    except Exception as exc:  # noqa: BLE001 - unknown name or privacy block, both user-facing
        raise PreflightError(str(exc)) from None
    if config.tts_backend == "kokoro":
        status = model_files_status()
        if status != "OK":
            raise PreflightError(status)

    return warnings
