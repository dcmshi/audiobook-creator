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
        raise RuntimeError(
            "ffmpeg/ffprobe not found on PATH. Install ffmpeg and retry; see 'abc doctor'."
        )
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
        # Track numbers are positional: chapter file numbering is sparse (front and back
        # matter are dropped), so int(stem)+1 would leave gaps in the album's track order.
        for track, (wav, (title, _dur)) in enumerate(zip(wavs, chapters, strict=True), start=1):
            mp3 = job.output_dir / f"{wav.stem} - {safe_filename(title)}.mp3"
            encode_mp3(
                wav,
                mp3,
                title=title,
                artist=doc.meta.author,
                album=doc.meta.title,
                track=track,
            )
    if "m4b" in formats:
        meta_path = job.output_dir / "ffmetadata.txt"
        write_ffmetadata(
            meta_path, title=doc.meta.title, artist=doc.meta.author, chapters=chapters
        )
        cover = Path(doc.meta.cover_path) if doc.meta.cover_path else None
        out = job.output_dir / f"{safe_filename(doc.meta.title)}.m4b"
        build_m4b(wavs, meta_path, out, cover)
