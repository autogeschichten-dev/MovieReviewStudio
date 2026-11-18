"""Exact-range video rendering and highlight-aware audio assembly."""
from dataclasses import replace
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import shutil
import tempfile
import wave

from .analysis import Analysis
from .media import Media, ffmpeg, identity, invoke, seconds, video_seek
from .model import Clip, validate

RATE = 48000
VERSION = "render-3"


def dimensions(movie: Media, preview: bool) -> tuple[int, int]:
    width, height = (640, 360) if preview else (1920, 1080)
    ratio = Fraction(movie.width, movie.height) * movie.sar
    if ratio > Fraction(width, height):
        return width, max(2, round(width / ratio / 2) * 2)
    return max(2, round(height * ratio / 2) * 2), height


def render_clip(movie: Media, clip: Clip, destination: Path, preview: bool):
    width, height = dimensions(movie, preview)
    ffmpeg(*video_seek(movie, clip.start), "-map", "0:v:0", "-an", "-sn", "-dn",
           "-vf", f"scale={width}:{height}:flags=lanczos,setsar=1,setpts=N*{movie.fps.denominator}/{movie.fps.numerator}/TB",
           "-frames:v", str(clip.frames), "-r", str(movie.fps), "-fps_mode", "cfr",
           "-c:v", "libx264", "-threads", "3", "-preset", "ultrafast" if preview else "fast",
           "-crf", "25" if preview else "17", "-pix_fmt", "yuv420p", str(destination))
    frames = invoke(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                     "stream=nb_frames", "-of", "csv=p=0", str(destination)]).strip()
    if frames != str(clip.frames):
        raise RuntimeError(f"Expected {clip.frames} frames at source frame {clip.start}, got {frames}.")


def audio_spans(clips: list[Clip]) -> list[tuple[bool, int, int]]:
    spans = []
    cursor = 0
    for clip in clips:
        end = cursor + clip.frames
        if spans and not clip.highlight and not spans[-1][0]:
            spans[-1] = (False, spans[-1][1], end)
        else:
            spans.append((clip.highlight, cursor, end))
        cursor = end
    return spans


def audio_track(movie: Media, clips: list[Clip], voice: Path | None, music: Path | None, folder: Path) -> Path:
    def samples(frame):
        return round(frame * RATE / movie.fps)

    # Normalize narration and one cycle of music for sample-accurate seeking.
    inputs = {}
    for name, path in (("voice", voice), ("music", music)):
        if path:
            normalized = folder / f"{name}.wav"
            ffmpeg("-i", str(path), "-map", "0:a:0", "-vn", "-ac", "2", "-ar", str(RATE), str(normalized))
            with wave.open(str(normalized), "rb") as source:
                if source.getnframes() == 0:
                    raise ValueError(f"The {name} track contains no audio samples.")
                inputs[name] = normalized, source.getnframes()
    originals = iter(clip for clip in clips if clip.highlight)
    bed_cursor = 0
    output = folder / "audio.wav"
    spans = audio_spans(clips)
    with wave.open(str(output), "wb") as result:
        result.setparams((2, 2, RATE, 0, "NONE", "not compressed"))
        for index, (highlight, start, end) in enumerate(spans):
            count = samples(end) - samples(start)
            target = folder / "span.wav"
            limit = f"apad=whole_len={count},atrim=end_sample={count}"
            if highlight:
                clip = next(originals)
                timestamp = max(0, movie.offset + float(clip.start / movie.fps))
                ffmpeg("-ss", seconds(timestamp), "-i", str(movie.path), "-map", "0:a:0", "-vn",
                       "-af", f"aresample={RATE}:async=1:first_pts=0,{limit}",
                       "-ac", "2", "-ar", str(RATE), str(target))
            else:
                args = []
                filters = []
                fade = min(RATE // 4, count // 2)
                for name, (path, length) in inputs.items():
                    if name == "voice" and bed_cursor >= length:
                        continue
                    offset = bed_cursor if name == "voice" else bed_cursor % length
                    if name == "music":
                        args += ["-stream_loop", "-1"]
                    args += ["-ss", seconds(Fraction(offset, RATE)), "-i", str(path)]
                    fade_ins = f",afade=t=in:ss=0:ns={fade}" if index else ""
                    fade_outs = f",afade=t=out:ss={count - fade}:ns={fade}" if index < len(spans) - 1 else ""
                    gain = "0.92" if name == "voice" else "0.16"
                    filters.append(f"[{len(filters)}:a]{limit},volume={gain}{fade_ins}{fade_outs}[{name}]")
                labels = "".join(f"[{name}]" for name in inputs if any(item.endswith(f"[{name}]") for item in filters))
                if filters:
                    filters.append(f"{labels}amix=inputs={len(filters)}:normalize=0[mix]")
                    ffmpeg(*args, "-filter_complex", ";".join(filters), "-map", "[mix]",
                           "-ac", "2", "-ar", str(RATE), str(target))
                else:
                    ffmpeg("-f", "lavfi", "-i", f"anullsrc=r={RATE}:cl=stereo", "-af", f"atrim=end_sample={count}",
                           str(target))
                bed_cursor += count
            with wave.open(str(target), "rb") as source:
                if source.getnframes() != count:
                    raise RuntimeError(f"Audio span {index + 1} has an unexpected length.")
                while block := source.readframes(RATE):
                    result.writeframesraw(block)
    return output


def render(movie: Media, analysis: Analysis, clips: list[Clip], voice: Path | None,
           music: Path | None, cache: Path, progress, *, destination: Path | None = None) -> Path:
    preview = destination is None
    validate(clips, movie.fps, analysis.frames, final=not preview, has_audio=movie.has_audio)
    identities = [identity(path) if path else None for path in (movie.path, voice, music)]
    key_data = (VERSION, identities, [(c.start, c.end, c.highlight) for c in clips])
    key = hashlib.sha256(json.dumps(key_data).encode()).hexdigest()
    destination = destination or cache / f"preview-{key}.mp4"
    if destination.resolve() in {path.resolve() for path in (movie.path, voice, music) if path}:
        raise ValueError("Export to a different file from the imported media.")
    if preview and destination.is_file():
        return destination
    movie = replace(movie, offset=analysis.offset)
    with tempfile.TemporaryDirectory(prefix=".mrs-", dir=destination.parent) as temporary:
        folder = Path(temporary)
        listing = []
        for index, clip in enumerate(clips, 1):
            progress(f"Rendering video clip {index} of {len(clips)}…")
            name = f"clip-{index:04}.mp4"
            part = folder / name
            chunk_key = f"{VERSION}-{identities[0]}-{clip.start}-{clip.end}.mp4"
            cached = cache / chunk_key
            if preview and cached.is_file():
                shutil.copyfile(cached, part)
            else:
                render_clip(movie, clip, part, preview)
                if preview:
                    pending = folder / "chunk.mp4"
                    shutil.copyfile(part, pending)
                    pending.replace(cached)
            listing.extend((f"file '{name}'", f"duration {seconds(clip.frames / movie.fps)}"))
        concat = folder / "clips.ffconcat"
        concat.write_text("\n".join(listing) + "\n", encoding="ascii")
        progress("Mixing narration, music and original-audio highlights…")
        audio = audio_track(movie, clips, voice, music, folder)
        intermediate = folder / "result.mp4"
        ffmpeg("-f", "concat", "-safe", "1", "-i", str(concat), "-i", str(audio),
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
               "-movflags", "+faststart", str(intermediate))
        if identities != [identity(path) if path else None for path in (movie.path, voice, music)]:
            raise ValueError("An imported file changed during rendering. Import the media again.")
        intermediate.replace(destination)
    return destination
