"""Sample-accurate processing and picture synchronization for imported narration."""
from bisect import bisect_right
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import wave

import numpy as np

from . import director, local_ai
from .dialogue import parse_captions
from .media import ffmpeg, identity, inspect
from .model import adjacent_conflict, validate
from .render import RATE, audio_spans
from .smart import audio_evidence

BLOCK = RATE // 100
QUIET = 104  # About -50 dBFS peak, measured on both channels.


@dataclass(frozen=True)
class Part:
    path: Path
    duration: float
    peaks: tuple
    stamp: str


@dataclass(frozen=True)
class Recording:
    path: Path
    frames: int
    # One entry per part: merged start, removed input head, and retained length, in samples.
    boundaries: tuple

    @property
    def duration(self):
        return self.frames / RATE


def pcm(path):
    return ffmpeg("-i", str(path), "-map", "0:a:0", "-vn", "-ac", "2", "-ar", str(RATE),
                  "-f", "s16le", "pipe:1", binary=True)


def read_part(path, cache, progress):
    path = Path(path).resolve()
    if path.suffix.lower() not in {".wav", ".mp3"}:
        raise ValueError("Choose a WAV or MP3 narration file.")
    stamp = identity(path)
    evidence = audio_evidence(path, cache, progress)
    if not evidence.peaks:
        raise ValueError(f"{path.name} contains no audio samples.")
    duration = inspect(path, video=False).duration
    if stamp != identity(path):
        raise ValueError(f"{path.name} changed during import. Try importing it again.")
    return Part(path, duration, evidence.peaks, stamp)


def writer(path):
    output = wave.open(str(path), "wb")
    output.setparams((2, 2, RATE, 0, "NONE", "not compressed"))
    return output


def quiet_blocks(samples):
    # Use peak in either channel so cancellation from a stereo downmix cannot hide speech.
    full = len(samples) // BLOCK * BLOCK
    blocks = samples[:full].reshape(-1, BLOCK, 2)
    return (blocks.max(axis=(1, 2)) <= QUIET) & (blocks.min(axis=(1, 2)) >= -QUIET)


def trim_edges(samples):
    active = np.flatnonzero(~quiet_blocks(samples))
    # Only trim sustained edge silence, retaining at least 150 ms and never trimming internal pauses.
    if not len(active):
        return 0, len(samples)
    first, last = int(active[0]) * BLOCK, (int(active[-1]) + 1) * BLOCK
    start = max(0, first - 15 * BLOCK) if first >= 30 * BLOCK else 0
    residual = samples[len(samples) // BLOCK * BLOCK:]
    end = (last + 15 * BLOCK if len(samples) - last >= 30 * BLOCK
           and (not len(residual) or np.abs(residual.astype(np.int32)).max() <= QUIET)
           else len(samples))
    return start, end


def gain_for(path, peak):
    if not peak:
        return 1
    output = ffmpeg("-i", str(path), "-af",
                    "ebur128=metadata=1,ametadata=print:key=lavfi.r128.I:file=-", "-f", "null", "-")
    measures = re.findall(r"lavfi\.r128\.I=(-?\d+(?:\.\d+)?)", output)
    measured = float(measures[-1]) if measures else -70
    gain = 10 ** (min(18, -18 - measured) / 20) if measured > -65 else 1
    return min(gain, .89 * 32768 / peak)


def merge(parts, pause_ms, trim, cache, progress):
    if not parts:
        raise ValueError("Import at least one narration part.")
    if not 0 <= pause_ms <= 5000:
        raise ValueError("Choose a pause between zero and five seconds.")
    sources = [part.stamp for part in parts]
    if sources != [identity(part.path) for part in parts]:
        raise ValueError("An imported narration file changed. Replace that part before merging.")
    key = sha256(json.dumps([2, sources, pause_ms, trim]).encode()).hexdigest()
    destination = cache / f"narration-{key}.wav"
    metadata = destination.with_suffix(".json")
    if destination.is_file() and metadata.is_file():
        frames, boundaries = json.loads(metadata.read_text(encoding="utf-8"))
        return Recording(destination, frames, tuple(map(tuple, boundaries)))
    pause = round(pause_ms * RATE / 1000)
    boundaries, cursor = [], 0
    with tempfile.TemporaryDirectory(prefix="mrs-narration-", dir=cache) as temporary:
        pending = Path(temporary) / "merged.wav"
        dry = Path(temporary) / "part.wav"
        with writer(pending) as output:
            for i, part in enumerate(parts):
                progress(f"Merging narration · {i + 1}/{len(parts)} · {part.path.name}")
                samples = np.frombuffer(pcm(part.path), dtype="<i2").reshape(-1, 2)
                if not len(samples):
                    raise ValueError(f"{part.path.name} contains no audio samples.")
                start, end = trim_edges(samples) if trim else (0, len(samples))
                selected = samples[start:end]
                with writer(dry) as item:
                    item.writeframes(selected.tobytes())
                gain = gain_for(dry, max(int(selected.max()), -int(selected.min())))
                if i and pause:
                    output.writeframesraw(bytes(pause * 4))
                    cursor += pause
                boundaries.append((cursor, start, len(selected)))
                cursor += len(selected)
                for offset in range(0, len(selected), RATE):
                    block = selected[offset:offset + RATE].astype(np.float32) * gain
                    # A two-millisecond de-click ramp never removes or overlaps recorded samples.
                    edge = min(96, len(selected) // 2)
                    if edge and offset == 0:
                        block[:edge] *= np.linspace(0, 1, edge)[:, None]
                    if edge and offset + len(block) == len(selected):
                        count = min(edge, len(block))
                        block[-count:] *= np.linspace(1, 0, edge)[-count:, None]
                    elif edge and len(selected) - offset - len(block) < edge:
                        remainder = len(selected) - offset - len(block)
                        block[-(edge - remainder):] *= np.linspace(1, 0, edge)[:edge - remainder, None]
                    output.writeframesraw(np.rint(block).astype("<i2").tobytes())
        if sources != [identity(part.path) for part in parts]:
            raise ValueError("A narration part changed during merging. Import it again.")
        pending.replace(destination)
        from .analysis import atomic_json
        atomic_json(metadata, [cursor, boundaries])
    return Recording(destination, cursor, tuple(boundaries))


def transcript(recording, parts, captions, cache, progress):
    if captions:
        cues = parse_captions(captions.read_text(encoding="utf-8-sig"))
        if not cues:
            raise ValueError("The selected narration subtitle file has no timestamped cues.")
        return cues
    cues = []
    for part, (position, head, length) in zip(parts, recording.boundaries):
        sidecar = next((part.path.with_suffix(ext) for ext in (".srt", ".vtt")
                        if part.path.with_suffix(ext).is_file()), None)
        if not sidecar:
            break
        part_cues = parse_captions(sidecar.read_text(encoding="utf-8-sig"))
        if not part_cues:
            break
        for cue in part_cues:
            start = max(0, cue["start"] - head / RATE)
            end = min(length / RATE, cue["end"] - head / RATE)
            if end > start:
                cues.append(dict(cue, start=start + position / RATE, end=end + position / RATE))
    else:
        return cues
    if local_ai.whisper_python():
        result = local_ai.dialogue(SimpleNamespace(path=recording.path, has_audio=True), cache, progress)
        return result["segments"]
    return director.read_transcript(SimpleNamespace(path=recording.path), cache, progress)


def quiet_ranges(recording):
    with wave.open(str(recording.path), "rb") as source:
        blocks = []
        while raw := source.readframes(RATE):
            blocks.extend(quiet_blocks(np.frombuffer(raw, dtype="<i2").reshape(-1, 2)).tolist())
    transitions = np.flatnonzero(np.diff(np.r_[False, blocks, False]))
    # Leave 40 ms of verified quiet on each side of any interruption.
    return [(int(a + 4) * BLOCK, int(b - 4) * BLOCK)
            for a, b in zip(transitions[::2], transitions[1::2]) if b - a >= 8]


def last_pause(quiet, lower, upper):
    index = bisect_right(quiet, upper, key=lambda interval: interval[0]) - 1
    if index >= 0 and lower <= upper and quiet[index][1] >= lower:
        return min(quiet[index][1], upper)
    return None


def sentence_blocks(clips, first, end, section, timing, cues, quiet, fps):
    """Refine a paragraph at supported sentence-to-scene anchors when the picture has room."""
    from .planning import tokens

    source = {(a, b): tokens(description) for a, b, description in section.evidence}
    start_time, end_time = timing
    local_cues = [cue for cue in cues if cue["start"] >= start_time and cue["end"] <= end_time]
    choices = []
    last_index, last_cut = 0, first
    for before, after in zip(local_cues, local_cues[1:]):
        if not before["text"].rstrip().endswith((".", "?", "!", "؟", "。")):
            continue
        cut = last_pause(quiet, math.ceil(before["end"] * RATE), math.floor(after["start"] * RATE))
        if cut is None or not last_cut < cut < end:
            continue
        words = tokens(after["text"])
        for index, clip in enumerate(clips[last_index + 1:], last_index + 1):
            if clip.movie_audio:
                continue
            shared = words & source.get((clip.start, clip.end), set())
            if len(shared) < 2:
                continue
            candidate = (clips[last_index:index], last_cut, cut)
            try:
                place([*choices, candidate, (clips[index:], cut, end)], fps, quiet)
            except ValueError:
                continue
            choices.append(candidate)
            last_index, last_cut = index, cut
            break
    return [*choices, (clips[last_index:], last_cut, end)]


def section_blocks(recording, clips, script, cues, quiet, fps):
    if not script or not cues:
        return [(list(clips), 0, recording.frames)], list(script), False
    timing = director.match_recording(script, cues)
    for start, end in timing.values():
        if not 0 <= start < end <= recording.duration + .05:
            raise ValueError("A matched narration cue lies outside the merged recording.")
    ends = []
    for left, right in zip(script, script[1:]):
        end = last_pause(quiet, math.ceil(timing[left.id][1] * RATE), math.floor(timing[right.id][0] * RATE))
        if end is None:
            raise ValueError(f"There's no verified pause between {left.id} and {right.id}. Check the narration cue timings.")
        ends.append(end)
    ends.append(recording.frames)
    owner = {(a, b): i for i, section in enumerate(script) for a, b, _ in section.evidence}
    blocks = [[] for _ in script]
    current = 0
    for clip in clips:
        if not clip.movie_audio:
            chapter = owner.get((clip.start, clip.end))
            if chapter is None or chapter < current:
                raise ValueError("Link each narration clip to its script section in timeline order before aligning.")
            current = chapter
        blocks[current].append(clip)
    points = [0, *ends]
    refined = [piece for section, block, a, b in zip(script, blocks, points, ends)
               for piece in sentence_blocks(block, a, b, section, timing[section.id], cues, quiet, fps)]
    return (refined, [replace(section, duration=(b - a) / RATE)
                      for section, a, b in zip(script, points, ends)], True)


def place(blocks, fps, quiet):
    """Place every sample once, consuming narration only under muted picture."""
    schedule = []
    picture = bed = 0
    for index, (clips, first, end) in enumerate(blocks, 1):
        spans = audio_spans(clips)
        cursor = first
        for original, start, stop in spans:
            if original:
                continue
            count = round((picture + stop) * RATE / fps) - round((picture + start) * RATE / fps)
            last = end if end - cursor <= count else last_pause(quiet, cursor, cursor + count)
            if last is None:
                last = cursor
            if last > cursor:
                schedule.append((bed, cursor, last))
            cursor = last
            bed += count
        if cursor != end:
            raise ValueError(f"Narration section {index} needs {(end - cursor) / RATE:.2f} s more usable picture or a verified "
                             "speech pause before an original-audio cut. Add relevant footage, move the highlight, "
                             "or adjust the pauses between parts.")
        picture += sum(clip.frames for clip in clips)
    return tuple(schedule), bed


def synchronize(recording, clips, script, cues, movie, analysis, min_gap, fit, progress):
    if not clips:
        raise ValueError("Generate or edit a montage before synchronizing narration.")
    validate(clips, movie.fps, analysis.frames, has_audio=movie.has_audio, min_gap=min_gap)
    quiet = quiet_ranges(recording)
    blocks, script, aligned = section_blocks(recording, clips, script, cues, quiet, movie.fps)
    # When fitting, remove only whole, surplus muted edits from their own section.
    if fit:
        for index, (block, start, end) in enumerate(blocks):
            progress(f"Fitting narration picture · {index + 1}/{len(blocks)}")
            for pos in range(len(block) - 1, -1, -1):
                clip = block[pos]
                if clip.movie_audio or clip.role:
                    continue
                left = block[pos - 1] if pos else next((row[0][-1] for row in reversed(blocks[:index]) if row[0]), None)
                right = (block[pos + 1] if pos + 1 < len(block) else
                         next((row[0][0] for row in blocks[index + 1:] if row[0]), None))
                if left and right and adjacent_conflict(left, right, movie.fps, min_gap):
                    continue
                block.pop(pos)
                try:
                    place(blocks, movie.fps, quiet)
                except ValueError:
                    block.insert(pos, clip)
    picture = [clip for block, _, _ in blocks for clip in block]
    validate(picture, movie.fps, analysis.frames, has_audio=movie.has_audio, min_gap=min_gap)
    schedule, length = place(blocks, movie.fps, quiet)
    return picture, script, schedule, length, aligned


def materialize(recording, schedule, length, cache):
    if not schedule or schedule[0][1] != 0 or schedule[-1][2] != recording.frames:
        raise ValueError("The saved narration mapping does not cover the recording.")
    previous_end = cursor = 0
    for position, start, end in schedule:
        if start != previous_end or not start < end or not cursor <= position <= position + end - start <= length:
            raise ValueError("The saved narration mapping contains a gap or overlap.")
        cursor, previous_end = position + end - start, end
    key = sha256(json.dumps([identity(recording.path), schedule, length]).encode()).hexdigest()
    destination = cache / f"narration-bed-{key}.wav"
    if destination.is_file():
        return destination
    with tempfile.TemporaryDirectory(prefix="mrs-narration-bed-", dir=cache) as folder:
        path = Path(folder) / "bed.wav"
        with writer(path) as output, wave.open(str(recording.path), "rb") as source:
            cursor = 0
            for position, start, end in (*schedule, (length, recording.frames, recording.frames)):
                while cursor < position:
                    count = min(RATE, position - cursor)
                    output.writeframesraw(bytes(count * 4))
                    cursor += count
                source.setpos(start)
                while start < end:
                    count = min(RATE, end - start)
                    output.writeframesraw(source.readframes(count))
                    start += count
                    cursor += count
        path.replace(destination)
    return destination


def markers(recording, schedule):
    return tuple((number, (position + start - first) / RATE)
                 for number, (start, _, _) in enumerate(recording.boundaries, 1)
                 for position, first, end in schedule if first <= start < end)
