"""Decode the entire source and index shot-contained edit candidates."""
from bisect import bisect_right
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
from .model import Clip, MAX_CLIP_SECONDS, Signature, adjacent_conflict, highlight_cuts, overlaps, validate

WIDTH, HEIGHT = 320, 180
FRAME_BYTES = WIDTH * HEIGHT * 3
VERSION = "analysis-6"


@dataclass(frozen=True)
class Analysis:
    frames: int
    offset: float
    candidates: list[Clip]
    shot_starts: tuple[int, ...] = ()


def thumbnail_path(cache: Path, source_key: str, frame: int) -> Path:
    return cache / f"{VERSION}-{source_key}-thumbs" / f"{frame}.jpg"


def save_thumbnail(cache: Path, source_key: str, index: int, frame: np.ndarray):
    path = thumbnail_path(cache, source_key, index)
    path.parent.mkdir(exist_ok=True)
    image = cv2.resize(frame, (128, 72), interpolation=cv2.INTER_AREA)
    encoded, data = cv2.imencode(".jpg", image)
    if encoded:
        path.write_bytes(data.tobytes())


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
                highlight=item["highlight"], shot=item["shot"], production_card=item.get("production_card", False))


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
                            [clip_from_dict(item) for item in payload["candidates"]],
                            tuple(payload["shot_starts"]))
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
    limit = int(MAX_CLIP_SECONDS * media.fps)
    minimum = max(1, round(float(media.fps) * .40))
    candidates = []
    fingerprints = []
    metrics = []
    previous = None
    first_frame = None
    start = 0
    shot = 0
    shot_starts = []
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    faces = cv2.CascadeClassifier(str(cascade_path)) if cascade_path.is_file() else None

    def finish():
        if len(fingerprints) >= minimum and metrics:
            quality, unusable, credits = np.mean(metrics, axis=0)
            if unusable < .32 and credits < (.45 if start > .85 * count else .70):
                gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
                face_count = len(faces.detectMultiScale(gray, 1.18, 4, minSize=(25, 25))) if faces is not None and not faces.empty() else 0
                card = start < min(count * .06, 90 * media.fps) and credits >= .40 and face_count == 0
                candidate = make_clip(start, fingerprints, float(quality) + .65 * min(face_count, 2), shot)
                candidates.append(replace(candidate, production_card=bool(card)))
                save_thumbnail(cache, key, start, first_frame)
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
                    shot_starts.append(index)
                if not black:
                    if not fingerprints:
                        start = index
                        first_frame = frame
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
    atomic_json(result_path, {"frames": count, "offset": offset, "candidates": [asdict(c) for c in candidates],
                              "shot_starts": shot_starts})
    return Analysis(count, offset, candidates, tuple(shot_starts))


def cutaway_without_visible_speaker(media, clip, detector):
    """Admit quiet-looking local scenery conservatively; never choose talking-head coverage."""
    if detector.empty():
        return False
    indices = sorted({0, clip.frames // 2, clip.frames - 1})
    select = "+".join(f"eq(n\\,{index})" for index in indices)
    data = ffmpeg(*video_seek(media, clip.start), "-map", "0:v:0", "-an",
                  "-vf", f"select={select},scale={WIDTH}:{HEIGHT}",
                  "-frames:v", str(len(indices)), "-fps_mode", "passthrough", "-pix_fmt", "bgr24",
                  "-f", "rawvideo", "-threads", "1", "pipe:1", binary=True)
    if len(data) != FRAME_BYTES * len(indices):
        return False
    for frame in np.frombuffer(data, dtype=np.uint8).reshape((-1, HEIGHT, WIDTH, 3)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(detector.detectMultiScale(gray, 1.1, 4, minSize=(18, 18))):
            return False
        _, _, black, unusable, text = measure(frame, None)
        if black or unusable or text > .55:
            return False
    return True


def source_fragment(media, analysis, start, end, cache):
    parent = next((c for c in analysis.candidates if c.start <= start < end <= c.end and c.shot is not None), None)
    if parent is None:
        raise ValueError("Picture range is not inside an analyzed source shot.")
    return parent if (start, end) == (parent.start, parent.end) else fingerprint_highlight(
        media, analysis, start, end, cache=cache, candidate=parent)


def highlight_picture(media, analysis, start, end, cache, reserved, min_gap, screened=None, *,
                      rejected=None, related=()):
    from collections import Counter
    from .model import cut_issue
    if rejected is None:
        rejected = Counter()
    if screened is None:
        screened = {}
    fps = media.fps
    needed = end - start
    media = replace(media, offset=analysis.offset)
    anchor = None
    nearby = []
    for candidate in analysis.candidates:
        distance = max(0, start - candidate.end, candidate.start - end)
        grounded = any(a < candidate.end and candidate.start < b for a, b in related)
        if candidate.production_card:
            rejected["production_card"] += 1
        elif candidate.shot is None:
            rejected["unknown_shot"] += 1
        elif any(overlaps(candidate, clip) for clip in reserved):
            rejected["reserved_picture"] += 1
        elif candidate.start <= start < candidate.end:
            anchor = source_fragment(media, analysis, start, min(end, candidate.end), cache)
        elif candidate.start < end and candidate.end > start:
            rejected["out_of_sync_passage_picture"] += 1
        elif distance > 180 * fps and not grounded:
            rejected["outside_context_without_grounding"] += 1
        else:
            nearby.append((distance, grounded, candidate))
    nearby.sort(key=lambda row: (row[0], not row[1], -row[2].frames, -row[2].score, row[2].start))
    detector = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    for seconds in (45, 90, 180):
        pool = [c for distance, grounded, c in nearby if distance <= seconds * fps or grounded][:96]
        budget = 320

        def chain(picture, remaining):
            nonlocal budget
            if not remaining:
                return tuple(picture)
            for candidate in pool:
                if budget <= 0:
                    rejected["picture_search_budget"] += 1
                    break
                if any(candidate.shot == used.shot for used in picture):
                    rejected["reused_source_shot"] += 1
                    continue
                piece_end = min(candidate.end, candidate.start + remaining)
                approximate = replace(candidate, end=piece_end)
                if picture and (reason := cut_issue(picture[-1], approximate, fps, min_gap)):
                    rejected[reason] += 1
                    continue
                key = candidate.start, candidate.end
                if key not in screened:
                    screened[key] = cutaway_without_visible_speaker(media, candidate, detector)
                if not screened[key]:
                    rejected["visible_face_or_unusable_cutaway"] += 1
                    continue
                budget -= 1
                piece = source_fragment(media, analysis, candidate.start, piece_end, cache)
                if picture and (reason := cut_issue(picture[-1], piece, fps, min_gap)):
                    rejected[reason] += 1
                    continue
                result = chain([*picture, piece], remaining - piece.frames)
                if result:
                    return result
            return None

        # A synchronous anchor is useful, but can prevent the only complete strict chain.
        for prefix in ([anchor], []) if anchor else ([],):
            result = chain(prefix, needed - sum(c.frames for c in prefix))
            if result:
                return result
    raise ValueError("No suitable non-adjacent picture for this original-audio range. The allocation report lists the rejected cuts.")

def fingerprint_highlight(media: Media, analysis: Analysis, start: int, end: int, *,
                          cache: Path | None = None, cuts: tuple[int, ...] | None = None,
                          role: str = "", candidate: Clip | None = None, picture=None,
                          reserved=(), min_gap=5, screened=None, rejected=None, related=()) -> Clip:
    if candidate:
        if not candidate.start <= start < end <= candidate.end:
            raise ValueError("A fragment must stay inside its analyzed source shot.")
    else:
        minimum, maximum = (1, 30) if role in {"Opening", "Ending"} else (1, 15) if role == "Highlight" else (10, 15)
        if not media.has_audio or not 0 <= start < end <= analysis.frames or not minimum <= (end - start) / media.fps <= maximum:
            raise ValueError(f"An original-audio group needs a {minimum}–{maximum} second source range.")
        clip = Clip(start, end, 0, (), (), (), True, role=role)
        if cuts is not None and picture is None:
            # Open old projects with unresolved cuts visible. Every playback/render path audits these pieces.
            legacy_cuts = cuts or highlight_cuts(start, end, media.fps, analysis.shot_starts)
            return replace(clip, cuts=legacy_cuts, legacy_shots=tuple(
                bisect_right(analysis.shot_starts, point) for point in (start, *legacy_cuts)))
        fragments = (highlight_picture(media, analysis, start, end, cache, reserved, min_gap, screened,
                                       rejected=rejected, related=related) if picture is None
                     else tuple(Clip(row[0], row[1], 0, (), (), (), gap=True) if len(row) > 2 and row[2]
                                else source_fragment(media, analysis, row[0], row[1], cache) for row in picture))
        clip = replace(clip, picture=fragments)
        validate([clip], media.fps, analysis.frames, has_audio=media.has_audio, min_gap=min_gap)
        return clip

    media = replace(media, offset=analysis.offset)
    clip = replace(candidate, start=start, end=end)
    length = end - start
    middle = np.linspace(0, length - 1, min(5, length), dtype=int)
    edge = min(3, length)
    indices = sorted(set(range(edge)) | set(range(length - edge, length)) | set(middle))
    select = "+".join(f"eq(n\\,{index})" for index in indices)
    data = ffmpeg(*video_seek(media, start), "-map", "0:v:0", "-an",
                  "-vf", f"select={select},scale={WIDTH}:{HEIGHT}",
                  "-frames:v", str(len(indices)), "-fps_mode", "passthrough", "-pix_fmt", "bgr24",
                  "-f", "rawvideo", "-threads", "1", "pipe:1", binary=True)
    if len(data) != FRAME_BYTES * len(indices):
        raise RuntimeError("The decoder could not read the requested source fragment.")
    frames = np.frombuffer(data, dtype=np.uint8).reshape((-1, HEIGHT, WIDTH, 3))
    fingerprints = {index: signature(frame) for index, frame in zip(indices, frames)}
    if cache:
        save_thumbnail(cache, identity(media.path), start, frames[0])
    return replace(clip, first=tuple(fingerprints[i] for i in range(edge)),
                   last=tuple(fingerprints[i] for i in range(length - edge, length)),
                   samples=tuple(fingerprints[i] for i in middle))
