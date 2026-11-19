"""FFmpeg integration and media identity."""
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import logging
import subprocess
import sys


@dataclass(frozen=True)
class Media:
    path: Path
    width: int
    height: int
    fps: Fraction
    duration: float
    has_audio: bool
    offset: float
    sar: Fraction = Fraction(1)


def tool_path(name: str) -> str:
    if getattr(sys, "frozen", False):
        return str(Path(sys._MEIPASS) / "ffmpeg-bin" / f"{name}.exe")
    return name


def invoke(args: list[str], *, binary: bool = False):
    process = subprocess.run(
        [tool_path(args[0]), *args[1:]], capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    if process.returncode:
        message = process.stderr.decode("utf-8", "replace").strip()
        logging.error("%s failed: %s", args[0], message)
        raise RuntimeError(message.splitlines()[-1] if message else f"{args[0]} exited with {process.returncode}.")
    return process.stdout if binary else process.stdout.decode("utf-8", "replace")


def inspect(path: Path, *, video: bool = True) -> Media:
    data = json.loads(invoke(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))
    fmt = data["format"]
    has_audio = any(stream["codec_type"] == "audio" for stream in data["streams"])
    if not video:
        if not has_audio:
            raise ValueError("The file contains no audio stream.")
        return Media(path, 0, 0, Fraction(0), float(fmt.get("duration", 0)), True, 0)
    stream = next((item for item in data["streams"]
                   if item["codec_type"] == "video" and not item.get("disposition", {}).get("attached_pic")), None)
    if stream is None or stream != next(item for item in data["streams"] if item["codec_type"] == "video"):
        raise ValueError("The first video stream must contain the movie.")
    if any(item.get("rotation", 0) for item in stream.get("side_data_list", [])):
        raise ValueError("Rotate the source into its display orientation before importing.")
    fps = Fraction(stream["r_frame_rate"])
    if not 1 <= fps <= 120:
        raise ValueError("The movie needs a supported video frame rate between 1 and 120 FPS.")
    offset = float(stream.get("start_time", 0)) - float(fmt.get("start_time", 0))
    duration = float(stream.get("duration", fmt.get("duration", 0)))
    sar = stream.get("sample_aspect_ratio", "1:1")
    return Media(path, stream["width"], stream["height"], fps, duration, has_audio, offset,
                 Fraction(sar.replace(":", "/")) if sar not in ("N/A", "0:1") else Fraction(1))


def identity(path: Path) -> str:
    stat = path.stat()
    return hashlib.sha256(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()


def frame_clock(media: Media) -> tuple[int, float]:
    output = invoke(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                     "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(media.path)])
    timestamps = []
    for line in output.splitlines():
        value = line.split(",", 1)[0]
        if value:
            timestamps.append(float(value))
    if not timestamps:
        raise ValueError("The movie has no timestamped frames.")
    origin = timestamps[0]
    tolerance = min(0.002, 0.2 / float(media.fps))
    for index, timestamp in enumerate(timestamps):
        if abs(timestamp - origin - float(index / media.fps)) > tolerance:
            raise ValueError("The movie uses variable frame timing. Convert it to constant frame rate before importing.")
    return len(timestamps), origin


def ffmpeg(*args: str, binary: bool = False):
    return invoke(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                   "-filter_threads", "2", "-filter_complex_threads", "2", *args], binary=binary)


def seconds(value) -> str:
    return f"{float(value):.9f}"


def video_seek(media: Media, frame: int) -> list[str]:
    # Seek into the gap before the requested frame to avoid timestamp rounding.
    start = max(0, media.offset + float(Fraction(2 * frame - 1, 2) / media.fps))
    return ["-ss", seconds(start), "-threads", "2", "-i", str(media.path)]
