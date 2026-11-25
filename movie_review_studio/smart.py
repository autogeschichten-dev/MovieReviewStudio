"""Local audio evidence and chronological review selection."""
from bisect import bisect_left
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np

from .analysis import atomic_json, fingerprint_highlight
from .media import ffmpeg, identity, invoke
from .model import DEFAULT_SOURCE_GAP, adjacent_conflict, overlaps, similar, validate

PEAKS_PER_SECOND = 20


@dataclass(frozen=True)
class AudioEvidence:
    peaks: tuple[float, ...] = ()
    activity: tuple[float, ...] = ()
    captions: tuple[tuple[float, float], ...] = ()

    @property
    def duration(self):
        return len(self.peaks) / PEAKS_PER_SECOND

    def score(self, start, end):
        left = max(0, int(start * PEAKS_PER_SECOND))
        right = max(left + 1, int(end * PEAKS_PER_SECOND))
        active = self.activity[left:right]
        captioned = sum(max(0, min(end, last) - max(start, first)) for first, last in self.captions)
        return (sum(active) / max(1, len(active))) + min(1, captioned / max(.1, end - start))


def caption_ranges(content):
    def stamp(parts):
        hour, minute, second, millis = parts
        return int(hour or 0) * 3600 + int(minute) * 60 + int(second) + int(millis) / 1000

    clock = r'(?:(\d+):)?(\d\d):(\d\d)[,.](\d{3})'
    return tuple((stamp(match[:4]), stamp(match[4:])) for match in
                 re.findall(clock + r'\s+-->\s+' + clock, content))


def audio_evidence(path, cache, progress, *, movie=False, transcriber=None):
    subtitle = next((path.with_suffix(suffix) for suffix in ('.srt', '.vtt')
                     if path.with_suffix(suffix).is_file()), None) if movie else None
    key = identity(path)
    stamp = identity(subtitle) if subtitle else 'embedded' if movie else 'audio'
    cached = cache / f'evidence-1-{key}-{stamp}.json'
    if cached.is_file() and transcriber is None:
        peaks, activity, captions = json.loads(cached.read_text(encoding='utf-8'))
        return AudioEvidence(tuple(peaks), tuple(activity), tuple(map(tuple, captions)))
    progress(f'Reading audio waveform · {path.name}')
    raw = ffmpeg('-i', str(path), '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '8000',
                 '-f', 's16le', 'pipe:1', binary=True)
    samples = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768
    step = 8000 // PEAKS_PER_SECOND
    if len(samples) % step:
        samples = np.pad(samples, (0, step - len(samples) % step))
    windows = samples.reshape(-1, step)
    peaks = tuple(np.round(np.max(np.abs(windows), axis=1), 3).tolist())
    rms = np.sqrt(np.mean(windows ** 2, axis=1))
    crossings = np.mean(np.signbit(windows[:, 1:]) != np.signbit(windows[:, :-1]), axis=1)
    activity = tuple((np.clip(rms * 12, 0, 1) * ((crossings > .02) & (crossings < .25))).round(2).tolist())
    captions = ()
    if movie:
        if subtitle:
            captions = caption_ranges(subtitle.read_text(encoding='utf-8-sig', errors='replace'))
        else:
            streams = json.loads(invoke(['ffprobe', '-v', 'error', '-select_streams', 's',
                                         '-show_entries', 'stream=index,codec_name', '-of', 'json', str(path)]))
            text_codecs = {'subrip', 'ass', 'ssa', 'webvtt', 'mov_text', 'text'}
            for stream in streams.get('streams', []):
                if stream.get('codec_name') in text_codecs:
                    try:
                        text = ffmpeg('-i', str(path), '-map', f"0:{stream['index']}", '-f', 'srt', 'pipe:1')
                        captions = caption_ranges(text)
                    except RuntimeError:
                        continue
                    if captions:
                        break
        if not captions and transcriber:
            progress('Transcribing dialogue locally…')
            captions = transcriber(path, progress)
    if transcriber is None:
        atomic_json(cached, [peaks, activity, captions])
    return AudioEvidence(peaks, activity, captions)


class LocalWhisper:
    """Adapter for an explicitly supplied whisper.cpp CLI and local model."""
    def __init__(self, executable: Path, model: Path):
        self.executable = executable
        self.model = model

    def __call__(self, path, progress):
        with tempfile.TemporaryDirectory(prefix='mrs-speech-') as temporary:
            root = Path(temporary)
            audio = root / 'dialogue.wav'
            ffmpeg('-i', str(path), '-map', '0:a:0', '-ac', '1', '-ar', '16000', str(audio))
            output = root / 'dialogue'
            with subprocess.Popen([str(self.executable), '-m', str(self.model), '-f', str(audio),
                                   '-l', 'auto', '-osrt', '-of', str(output)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as process:
                while True:
                    try:
                        if process.wait(timeout=1):
                            raise RuntimeError('Local transcription failed. Check the CLI and model.')
                        break
                    except subprocess.TimeoutExpired:
                        try:
                            progress('Transcribing dialogue locally…')
                        except Exception:
                            process.kill()
                            raise
            return caption_ranges(output.with_suffix('.srt').read_text(encoding='utf-8'))


def suggest_highlights(movie, analysis, evidence, cache, progress):
    if not movie.has_audio or not evidence.peaks:
        return []
    length = math.ceil(12 * movie.fps)
    candidates = [clip for clip in analysis.candidates if not clip.production_card]
    options = []
    for index, first in enumerate(candidates):
        end = first.end
        for following in candidates[index + 1:index + 40]:
            if following.start != end or end - first.start >= length:
                break
            end = following.end
        if end - first.start < length:
            continue
        end = first.start + length
        seconds = float(first.start / movie.fps)
        end_seconds = float(end / movie.fps)
        if any(a < point < b for a, b in evidence.captions for point in (seconds, end_seconds)):
            continue
        score = evidence.score(seconds + analysis.offset, end_seconds + analysis.offset)
        if score > .035:
            options.append((score, first.start, end))
    selected = []
    for _, start, end in sorted(options, reverse=True):
        if any(abs(start - clip.start) < analysis.frames / 6 for clip in selected):
            continue
        progress(f'Reading suggested highlight {len(selected) + 1}…')
        selected.append(fingerprint_highlight(movie, analysis, start, end, cache=cache))
        if len(selected) == 3:
            break
    return sorted(selected, key=lambda clip: clip.start)


def assemble(candidates, highlights, fps, source_frames, target_seconds, evidence, progress, *,
             min_gap=DEFAULT_SOURCE_GAP):
    """Distribute selected footage over the source and keep its source order."""
    minimum = math.ceil(1020 * fps)
    ceiling = int(1080 * fps)
    target = max(minimum, min(ceiling, round(target_seconds * fps)))
    selected = sorted(highlights, key=lambda clip: clip.start)
    starts = [clip.start for clip in selected]
    frames = sum(clip.frames for clip in selected)
    lanes = [[] for _ in range(16)]
    seen = {}

    def center(clip):
        return clip.samples[len(clip.samples) // 2] if clip.samples else None
    for clip in candidates:
        if clip.production_card:
            continue
        if not any(overlaps(clip, highlight) for highlight in highlights):
            lanes[min(15, clip.start * 16 // source_frames)].append(clip)
    for lane in lanes:
        lane.sort(key=lambda clip: clip.score + 1.5 * evidence.score(float(clip.start / fps),
                                                                      float(clip.end / fps)))
    prefer_distinct = True
    while frames < target:
        changed = False
        for lane in lanes:
            for index in range(len(lane) - 1, -1, -1):
                clip = lane[index]
                if frames + clip.frames > min(ceiling, target + int(5 * fps)):
                    continue
                sample = center(clip)
                if prefer_distinct and sample and any(similar(sample, used) for used in seen.get(sample.phash, ())):
                    continue
                slot = bisect_left(starts, clip.start)
                left = selected[slot - 1] if slot else None
                right = selected[slot] if slot < len(selected) else None
                if ((left and (overlaps(left, clip) or adjacent_conflict(left, clip, fps, min_gap))) or
                    (right and (overlaps(right, clip) or adjacent_conflict(clip, right, fps, min_gap)))):
                    continue
                selected.insert(slot, clip)
                starts.insert(slot, clip.start)
                frames += clip.frames
                if sample:
                    seen.setdefault(sample.phash, []).append(sample)
                lane.pop(index)
                changed = True
                break
            if frames >= target:
                break
        progress(f'Selecting chronological footage · {frames / float(fps):.0f} / {target_seconds} seconds')
        if not changed:
            if not prefer_distinct:
                break
            prefer_distinct = False
    if frames < minimum or frames < target - math.ceil(5 * fps):
        raise ValueError('There is insufficient distinct usable footage for the chosen 17–18 minute target and source separation.')
    validate(selected, fps, source_frames, final=True, min_gap=min_gap)
    return selected
