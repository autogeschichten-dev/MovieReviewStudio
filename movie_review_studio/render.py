"""Exact-range video rendering and highlight-aware audio assembly."""
from dataclasses import dataclass, replace
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import shutil
import tempfile
import wave

from .analysis import Analysis
from .media import Media, ffmpeg, identity, invoke, seconds, video_seek
from .model import DEFAULT_SOURCE_GAP, Clip, MAX_CLIP_SECONDS, timeline_entries, validate

RATE = 48000
VERSION = "render-6"


@dataclass(frozen=True)
class Mix:
    original: float = 1.0
    voice: float = .92
    music: float = .24


def dimensions(movie: Media, preview: bool) -> tuple[int, int]:
    width, height = (640, 360) if preview else (1920, 1080)
    ratio = Fraction(movie.width, movie.height) * movie.sar
    if ratio > Fraction(width, height):
        return width, max(2, round(width / ratio / 2) * 2)
    return max(2, round(height * ratio / 2) * 2), height


def render_clip(movie: Media, clip: Clip, destination: Path, preview: bool, *, fade=False):
    if not 0 < clip.frames / movie.fps <= MAX_CLIP_SECONDS:
        raise ValueError("Each video segment must be at most 5 seconds.")
    width, height = dimensions(movie, preview)
    fade_frames = min(clip.frames - 1, round(movie.fps))
    fading = f",fade=t=out:s={clip.frames - fade_frames - 1}:n={fade_frames}" if fade and fade_frames else ""
    ffmpeg(*video_seek(movie, clip.start), "-map", "0:v:0", "-an", "-sn", "-dn",
           "-vf", f"scale={width}:{height}:flags=lanczos,setsar=1,setpts=N*{movie.fps.denominator}/{movie.fps.numerator}/TB{fading}",
           "-frames:v", str(clip.frames), "-r", str(movie.fps), "-fps_mode", "cfr",
           "-c:v", "libx264", "-threads", "3", "-preset", "ultrafast" if preview else "fast",
           "-crf", "25" if preview else "17", "-pix_fmt", "yuv420p",
           # Independently encoded clips need identical decode/display order at every join.
           "-bf", "0",
           "-video_track_timescale", str(movie.fps.numerator), str(destination))
    frames = invoke(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                     "stream=nb_frames", "-of", "csv=p=0", str(destination)]).strip()
    if frames != str(clip.frames):
        raise RuntimeError(f"Expected {clip.frames} frames at source frame {clip.start}, got {frames}.")


def audio_spans(clips: list[Clip]) -> list[tuple[bool, int, int]]:
    spans = []
    cursor = 0
    for clip in clips:
        end = cursor + clip.frames
        if spans and not clip.movie_audio and not spans[-1][0]:
            spans[-1] = (False, spans[-1][1], end)
        else:
            spans.append((clip.movie_audio, cursor, end))
        cursor = end
    return spans


def audio_track(movie: Media, clips: list[Clip], voice: Path | None, music: Path | None, folder: Path, mix: Mix | None = None, progress=lambda _: None, *, managed_voice=False) -> Path:
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
    originals = iter(clip for clip in clips if clip.movie_audio)
    bed_cursor = 0
    output = folder / "audio.wav"
    spans = audio_spans(clips)
    if managed_voice:
        required = sum(samples(end) - samples(start) for original, start, end in spans if not original)
        if "voice" not in inputs or inputs["voice"][1] != required:
            raise ValueError("The synchronized narration does not match this edit. Run Merge & Synchronize again.")
    with wave.open(str(output), "wb") as result:
        result.setparams((2, 2, RATE, 0, "NONE", "not compressed"))
        for index, (highlight, start, end) in enumerate(spans):
            progress(f"Mixing audio region {index + 1} of {len(spans)}…")
            count = samples(end) - samples(start)
            target = folder / "span.wav"
            limit = f"apad=whole_len={count},atrim=end_sample={count}"
            if highlight:
                clip = next(originals)
                timestamp = max(0, movie.offset + float(clip.start / movie.fps))
                fade = min(round(.012 * RATE), count // 2)
                edges = f"afade=t=in:ss=0:ns={fade},afade=t=out:ss={count - fade}:ns={fade}"
                # Decode dialogue once across all the group's video cuts.
                original = f"aresample={RATE}:async=1:first_pts=0,{limit},{edges}"
                args = ["-ss", seconds(timestamp), "-i", str(movie.path)]
                if mix and "music" in inputs and clip.role != "Ending":
                    path, length = inputs["music"]
                    args += ["-stream_loop", "-1", "-ss", seconds(Fraction(samples(start) % length, RATE)),
                             "-i", str(path)]
                    filters = (f"[0:a:0]{original},volume={mix.original}[dialogue];"
                               f"[1:a:0]{limit},{edges},volume={mix.music * .18}[music];"
                               "[dialogue][music]amix=inputs=2:normalize=0,alimiter=limit=0.98:level=false:latency=true[mix]")
                    ffmpeg(*args, "-filter_complex", filters, "-map", "[mix]", "-ac", "2", "-ar", str(RATE), str(target))
                else:
                    gain = mix.original if mix else 1
                    ffmpeg(*args, "-map", "0:a:0", "-vn", "-af", f"{original},volume={gain}",
                           "-ac", "2", "-ar", str(RATE), str(target))
            else:
                args = []
                filters = []
                fade = min(RATE // 4, count // 2)
                for name, (path, length) in inputs.items():
                    if name == "voice" and bed_cursor >= length:
                        continue
                    offset = bed_cursor if name == "voice" else (samples(start) if mix else bed_cursor) % length
                    if name == "music":
                        args += ["-stream_loop", "-1"]
                    args += ["-ss", seconds(Fraction(offset, RATE)), "-i", str(path)]
                    fade_ins = f",afade=t=in:ss=0:ns={fade}" if index else ""
                    fade_outs = f",afade=t=out:ss={count - fade}:ns={fade}" if index < len(spans) - 1 else ""
                    if name == "voice" and managed_voice:
                        fade_ins = fade_outs = ""
                    gain = "0.92" if name == "voice" else "0.16"
                    if mix:
                        gain = mix.voice if name == "voice" else mix.music
                        if name == "music":
                            narration_samples = min(count, max(0, inputs.get("voice", (None, 0))[1] - bed_cursor))
                            if narration_samples:
                                gain = f"'if(lt(t,{narration_samples / RATE}),{gain * .5},{gain})':eval=frame"
                    filters.append(f"[{len(filters)}:a]{limit},volume={gain}{fade_ins}{fade_outs}[{name}]")
                labels = "".join(f"[{name}]" for name in inputs if any(item.endswith(f"[{name}]") for item in filters))
                if filters:
                    limiter = ",alimiter=limit=0.98:level=false:latency=true" if mix else ""
                    filters.append(f"{labels}amix=inputs={len(filters)}:normalize=0{limiter}[mix]")
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


def preview_chunk(cache: Path, source_key: str, clip: Clip, *, fade=False) -> Path:
    return cache / f"{VERSION}-{source_key}-{clip.start}-{clip.end}{'-fade' if fade else ''}.mp4"


def render_selection(movie: Media, analysis: Analysis, clips: list[Clip], row: int,
                     voice: Path | None, music: Path | None, cache: Path, progress, *, mix: Mix | None = None, managed_voice=False, fade_ending=False) -> Path:
    entry = timeline_entries(clips, movie.fps)[row]
    assembly = render(movie, analysis, clips, voice, music, cache, progress, mix=mix, managed_voice=managed_voice, fade_ending=fade_ending)
    destination = cache / f"{assembly.stem}-clip-{row + 1}.mp4"
    if destination.is_file():
        return destination
    start = round(entry.position * RATE / movie.fps)
    end = round((entry.position + entry.segment.frames) * RATE / movie.fps)
    with tempfile.TemporaryDirectory(prefix=".mrs-", dir=cache) as temporary:
        folder = Path(temporary)
        fade = fade_ending and row == len(timeline_entries(clips, movie.fps)) - 1 and entry.segment.role == "Ending"
        video = preview_chunk(cache, identity(movie.path), entry.segment, fade=fade)
        if not video.is_file():
            video = folder / "video.mp4"
            render_clip(replace(movie, offset=analysis.offset), entry.segment, video, True, fade=fade)
        result = folder / "selected.mp4"
        progress("Preparing the selected clip with its montage audio…")
        ffmpeg("-i", str(video), "-i", str(assembly), "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "copy", "-af", f"atrim=start_sample={start}:end_sample={end},asetpts=PTS-STARTPTS",
               "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", str(result))
        result.replace(destination)
    return destination


def render(movie: Media, analysis: Analysis, clips: list[Clip], voice: Path | None,
           music: Path | None, cache: Path, progress, *, destination: Path | None = None, mix: Mix | None = None,
           min_gap=DEFAULT_SOURCE_GAP, managed_voice=False, fitted_duration=False, fade_ending=False) -> Path:
    preview = destination is None
    if any(c.gap for c in clips):
        raise ValueError("Fill unresolved picture gaps before rendering.")
    validate(clips, movie.fps, analysis.frames, final=not preview and not fitted_duration, has_audio=movie.has_audio,
             min_gap=min_gap, enforce_separation=not preview)
    entries = timeline_entries(clips, movie.fps)
    identities = [identity(path) if path else None for path in (movie.path, voice, music)]
    key_data = (VERSION, identities, [(c.start, c.end, c.highlight, c.cuts, c.original, c.muted, c.role) for c in clips],
                (mix.original, mix.voice, mix.music) if mix else None)
    if managed_voice:
        key_data += ("managed-narration",)
    if fade_ending:
        key_data += ("fade-ending",)
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
        previous_us = 0
        for index, entry in enumerate(entries, 1):
            clip = entry.segment
            progress(f"Rendering video clip {index} of {len(entries)}…")
            name = f"clip-{index:04}.mp4"
            part = folder / name
            fade = fade_ending and index == len(entries) and clip.role == "Ending"
            cached = preview_chunk(cache, identities[0], clip, fade=fade)
            if preview and cached.is_file():
                shutil.copyfile(cached, part)
            else:
                render_clip(movie, clip, part, preview, fade=fade)
                if preview:
                    pending = folder / "chunk.mp4"
                    shutil.copyfile(part, pending)
                    pending.replace(cached)
            end_us = round((entry.position + clip.frames) * 1_000_000 / movie.fps)
            listing.extend((f"file '{name}'", f"duration {seconds(Fraction(end_us - previous_us, 1_000_000))}"))
            previous_us = end_us
        concat = folder / "clips.ffconcat"
        concat.write_text("\n".join(listing) + "\n", encoding="ascii")
        progress("Mixing narration, music and original-audio highlights…")
        audio = audio_track(movie, clips, voice, music, folder, mix, progress, managed_voice=managed_voice)
        intermediate = folder / "result.mp4"
        ffmpeg("-f", "concat", "-safe", "1", "-i", str(concat), "-i", str(audio),
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
               "-video_track_timescale", str(movie.fps.numerator), "-movflags", "+faststart", str(intermediate))
        if identities != [identity(path) if path else None for path in (movie.path, voice, music)]:
            raise ValueError("An imported file changed during rendering. Import the media again.")
        progress("Finishing output…")
        intermediate.replace(destination)
    return destination
