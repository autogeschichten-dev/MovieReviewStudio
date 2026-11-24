"""Local subtitle discovery and optional OCR for dialogue evidence."""
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import tempfile

from .media import ffmpeg, invoke
from .recap_upstream._common import run_cmd

TIME = re.compile(r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})")


def parse_captions(text):
    def seconds(value):
        h, m, s, ms = value.groups()
        return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    found = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "").lstrip("\ufeff")):
        lines = block.splitlines()
        for index, line in enumerate(lines):
            if "-->" not in line:
                continue
            times = TIME.findall(line)
            if len(times) != 2:
                break
            left, right = map(seconds, TIME.finditer(line))
            caption = re.sub(r"<[^>]*>", "", " ".join(lines[index + 1:])).strip()
            if right > left >= 0 and caption:
                found.append({"start": left, "end": right, "text": caption})
            break
    return sorted(found, key=lambda cue: cue["start"])


def subtitles(path, sidecar):
    if sidecar:
        cues = parse_captions(sidecar.read_text(encoding="utf-8-sig", errors="replace"))
        if cues:
            return {"provider": f"subtitle file ({sidecar.name})", "language": None, "segments": cues}
    metadata = json.loads(invoke(["ffprobe", "-v", "error", "-select_streams", "s", "-show_streams",
                                  "-of", "json", str(path)]))
    streams = metadata.get("streams", [])
    text_types = {"subrip", "ass", "ssa", "webvtt", "mov_text", "text"}
    supported = [stream for stream in streams if stream.get("codec_name") in text_types]
    supported.sort(key=lambda stream: stream.get("tags", {}).get("language", "").lower() not in {"en", "eng"})
    for stream in supported:
        try:
            cues = parse_captions(ffmpeg("-i", str(path), "-map", f"0:{stream['index']}", "-f", "srt", "pipe:1"))
        except RuntimeError:
            continue
        if cues:
            return {"provider": f"embedded subtitles (stream {stream['index']})",
                    "language": stream.get("tags", {}).get("language"), "segments": cues}
    return {"provider": "none", "language": None, "segments": [], "subtitle_streams": len(streams)}


def ocr_captions(movie, progress):
    executable = shutil.which("tesseract")
    if not executable:
        raise RuntimeError("Burned-in subtitle OCR requires Tesseract with English data on PATH.")
    cues = []
    interval = 2
    with tempfile.TemporaryDirectory(prefix="mrs-ocr-") as folder:
        frame = Path(folder) / "subtitle.png"
        for stamp in range(0, int(movie.duration), interval):
            if stamp % 20 == 0:
                progress(f"Dialogue OCR · {stamp}/{int(movie.duration)} seconds sampled")
            ffmpeg("-ss", str(stamp), "-i", str(movie.path), "-vf", "crop=iw:ih/4:0:ih*3/4,scale=1000:-2",
                   "-frames:v", "1", str(frame))
            result = run_cmd([executable, str(frame), "stdout", "-l", "eng", "--psm", "6", "tsv"], timeout=15)
            if result.returncode:
                raise RuntimeError("Tesseract could not read English subtitles. Check installed language data.")
            words = []
            for row in result.stdout.splitlines()[1:]:
                cells = row.split("\t")
                if len(cells) == 12 and cells[0] == "5" and float(cells[10]) >= 70 and cells[11].strip():
                    words.append(cells[11].strip())
            if len(words) >= 3:
                text = " ".join(words)
                if cues and cues[-1]["text"] == text and cues[-1]["end"] == stamp:
                    cues[-1]["end"] = min(movie.duration, stamp + interval)
                else:
                    cues.append({"start": stamp, "end": min(movie.duration, stamp + interval), "text": text})
        counts = Counter(cue["text"] for cue in cues)
        cues = [cue for cue in cues if counts[cue["text"]] <= 12]
    return {"provider": "possible burned-in subtitles (English OCR, approximate)" if cues else "none",
            "language": "en" if cues else None, "segments": cues}


def coverage(cues, duration):
    end = 0
    covered = 0
    for start, finish in sorted((max(0, cue["start"]), min(duration, cue["end"])) for cue in cues):
        covered += max(0, finish - max(start, end))
        end = max(end, finish)
    return 100 * covered / duration if duration else 0
