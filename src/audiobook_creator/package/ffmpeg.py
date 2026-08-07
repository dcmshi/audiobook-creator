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


def encode_mp3(
    wav: Path, mp3: Path, *, title: str, artist: str | None, album: str, track: int
) -> None:
    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(wav),
        "-c:a",
        "libmp3lame",
        "-q:a",
        "4",
        "-metadata",
        f"title={title}",
        "-metadata",
        f"album={album}",
        "-metadata",
        f"track={track}",
    ]
    if artist:
        args += ["-metadata", f"artist={artist}"]
    args.append(str(mp3))
    _run(args)


def _escape_meta(value: str) -> str:
    # FFMETADATA escapes: '=', ';', '#', '\' and newline
    return re.sub(r"([=;#\\\n])", r"\\\1", value)


def write_ffmetadata(
    path: Path, *, title: str, artist: str | None, chapters: list[tuple[str, float]]
) -> None:
    lines = [";FFMETADATA1", f"title={_escape_meta(title)}"]
    if artist:
        lines.append(f"artist={_escape_meta(artist)}")
    offset_ms = 0
    for chapter_title, duration_s in chapters:
        end_ms = offset_ms + round(duration_s * 1000)
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={offset_ms}",
            f"END={end_ms}",
            f"title={_escape_meta(chapter_title)}",
        ]
        offset_ms = end_ms
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_m4b(wavs: list[Path], meta_path: Path, out: Path, cover: Path | None) -> None:
    list_path = out.parent / "concat.txt"
    escaped = [w.as_posix().replace("'", "'\\''") for w in wavs]
    list_path.write_text("".join(f"file '{p}'\n" for p in escaped), encoding="utf-8")
    args = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-i",
        str(meta_path),
    ]
    if cover is not None and cover.exists():
        args += [
            "-i",
            str(cover),
            "-map",
            "0:a",
            "-map",
            "2:v",
            "-c:v",
            "mjpeg",
            "-disposition:v:0",
            "attached_pic",
        ]
    else:
        args += ["-map", "0:a"]
    args += ["-map_metadata", "1", "-c:a", "aac", "-b:a", "64k", "-f", "mp4", str(out)]
    _run(args)
    list_path.unlink(missing_ok=True)


def probe_chapters(m4b: Path) -> list[str]:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_chapters", str(m4b)],
        check=True,
        capture_output=True,
        text=True,
    )
    return [c["tags"]["title"] for c in json.loads(result.stdout)["chapters"]]
