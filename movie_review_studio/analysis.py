"""Decode the entire source and index shot-contained edit candidates."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from scenedetect.detectors import ContentDetector

from .media import Media, ffmpeg, frame_clock, identity, invoke, tool_path, video_seek
from .model import Clip, Signature

WIDTH, HEIGHT = 320, 180
FRAME_BYTES = WIDTH * HEIGHT * 3
VERSION = "analysis-4"


@dataclass(frozen=True)
class Analysis:
    frames: int
    offset: float
    candidates: list[Clip]


def signature(frame: np.ndarray) -> Signature:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thumbnail = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(small.astype(np.float32))[:8, :8].flatten()
    bits = coefficients > np.median(coefficients[1:])
    return Signature(int.from_bytes(np.packbits(bits).tobytes(), "big"), thumbnail.tobytes().hex())


def make_clip(start, fingerprints, score, shot):
    positions = np.linspace(0, len(fingerprints) - 1, min(5, len(fingerprints)), dtype=int)
    return Clip(start, start + len(fingerprints), score, tuple(fingerprints[:3]),
                tuple(fingerprints[-3:]), tuple(fingerprints[index] for index in positions), shot=shot)


def measure(frame, previous):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark = np.mean(gray < 28)
    spread = np.std(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_32F).var()
    motion = float(cv2.absdiff(previous, gray).mean()) if previous is not None else 0
    # Edge structure around the thirds, fairly exposed frames, and moderate motion.
    edges = cv2.Canny(gray, 70, 150)
    composition = edges[30:150, 50:270].mean() / 255
    color = np.mean(frame.max(axis=2).astype(np.int16) - frame.min(axis=2))
    text_like = dark > .64 and color < 13 and np.mean(gray > 185) > .007
    black = np.mean(gray < 18) > .97
    unusable = black or spread < 8 or sharpness < 12
    quality = (min(sharpness / 450, 1) + min(spread / 55, 1)
               + min(composition * 10, 1) + max(0, 1 - abs(motion - 12) / 25))
    return gray, float(quality), black, unusable, text_like


def clip_from_dict(item):
    return Clip(item["start"], item["end"], item["score"],
                *(tuple(Signature(**entry) for entry in item[key]) for key in ("first", "last", "samples")),
                highlight=item["highlight"], shot=item["shot"])


def atomic_json(path: Path, payload):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(payload, stream, ensure_ascii=False)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def analyze(media: Media, cache: Path, progress) -> Analysis:
    key = identity(media.path)
    result_path = cache / f"{VERSION}-{key}.json"
    if result_path.exists():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            progress("Loaded cached shot analysis.")
            return Analysis(payload["frames"], payload["offset"],
                            [clip_from_dict(item) for item in payload["candidates"]])
        except (ValueError, KeyError, TypeError):
            progress("Rebuilding an unreadable analysis cache.")
    progress("Checking source frame timestamps…")
    count, origin = frame_clock(media)
    format_start = float(json.loads(invoke(
        ["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of", "json", str(media.path)]
    ))["format"].get("start_time", 0))
    offset = origin - format_start
    cv2.setNumThreads(2)
    detector = ContentDetector(threshold=25, min_scene_len=1)
    limit = int(5 * media.fps)
    minimum = max(1, round(float(media.fps) * .40))
    candidates = []
    fingerprints = []
    metrics = []
    previous = None
    start = 0
    shot = 0

    def finish():
        if len(fingerprints) >= minimum and metrics:
            quality, unusable, credits = np.mean(metrics, axis=0)
            if unusable < .32 and credits < (.45 if start > .85 * count else .70):
                candidates.append(make_clip(start, fingerprints, float(quality), shot))
        fingerprints.clear()
        metrics.clear()

    command = [tool_path("ffmpeg"), "-nostdin", "-v", "error", "-threads", "2", "-i", str(media.path),
               "-map", "0:v:0", "-an", "-sn", "-filter_threads", "2", "-vf", f"scale={WIDTH}:{HEIGHT}",
               "-fps_mode", "passthrough", "-pix_fmt", "bgr24", "-f", "rawvideo", "-threads", "1", "pipe:1"]
    with tempfile.TemporaryFile() as errors, subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=errors,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    ) as process:
        index = 0
        try:
            while raw := process.stdout.read(FRAME_BYTES):
                if len(raw) != FRAME_BYTES:
                    raise RuntimeError("Incomplete frame from the movie decoder.")
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
                cuts = detector.process_frame(index, frame)
                black = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) < 18) > .97
                if cuts or len(fingerprints) == limit or black:
                    finish()
                    previous = None
                if cuts:
                    shot += 1
                if not black:
                    if not fingerprints:
                        start = index
                    fingerprints.append(signature(frame))
                    if (index - start) % 4 == 0:
                        previous, quality, _, unusable, credits = measure(frame, previous)
                        metrics.append((quality, unusable, credits))
                index += 1
                if index % 240 == 0:
                    progress(f"Analyzing movie shots… {min(100, index * 100 // count)}%")
        finally:
            if sys.exc_info()[0] is not None and process.poll() is None:
                process.kill()
        if process.wait():
            errors.seek(0)
            raise RuntimeError(errors.read().decode("utf-8", "replace").strip() or "Movie decoding failed.")
    finish()
    if index != count:
        raise ValueError(f"The decoder returned {index} frames, but the source clock lists {count}.")
    if identity(media.path) != key:
        raise ValueError("The source changed during analysis. Import it again.")
    atomic_json(result_path, {"frames": count, "offset": offset, "candidates": [asdict(c) for c in candidates]})
    return Analysis(count, offset, candidates)


def fingerprint_highlight(media: Media, analysis: Analysis, start: int, end: int) -> Clip:
    if not media.has_audio or not 0 <= start < end <= analysis.frames or not 10 <= (end - start) / media.fps <= 15:
        raise ValueError("A highlight needs movie audio and a 10–15 second range inside the movie.")
    media = replace(media, offset=analysis.offset)
    length = end - start
    middle = np.linspace(0, length - 1, 5, dtype=int)
    indices = sorted(set(range(3)) | set(range(length - 3, length)) | set(middle))
    select = "+".join(f"eq(n\\,{index})" for index in indices)
    data = ffmpeg(*video_seek(media, start), "-map", "0:v:0", "-an",
                  "-vf", f"select={select},scale={WIDTH}:{HEIGHT}",
                  "-frames:v", str(len(indices)), "-fps_mode", "passthrough", "-pix_fmt", "bgr24",
                  "-f", "rawvideo", "-threads", "1", "pipe:1", binary=True)
    if len(data) != FRAME_BYTES * len(indices):
        raise RuntimeError("The decoder could not read the full highlight range.")
    frames = np.frombuffer(data, dtype=np.uint8).reshape((-1, HEIGHT, WIDTH, 3))
    fingerprints = {index: signature(frame) for index, frame in zip(indices, frames)}
    return Clip(start, end, 0, tuple(fingerprints[i] for i in range(3)),
                tuple(fingerprints[i] for i in range(length - 3, length)),
                tuple(fingerprints[i] for i in middle), True)
