"""Narration-driven planning and frame-accurate picture edits."""
from dataclasses import replace
from fractions import Fraction
from math import ceil

from .analysis import fingerprint_highlight
from .model import adjacent_conflict, overlaps, validate
from .narration import last_pause, quiet_ranges
from .render import RATE


def narration_frames(recording, fps):
    return ceil(Fraction(recording.frames, RATE) * fps)


def duration_totals(recording, clips, fps):
    groups = {"Opening": 0, "Highlight": 0, "Ending": 0}
    for clip in clips:
        if clip.movie_audio:
            role = clip.role if clip.role in ("Opening", "Ending") else "Highlight"
            groups[role] += clip.frames
    narrated = recording.duration if recording else 0
    return (narrated, *(float(frames / fps) for frames in groups.values()),
            narrated + float(sum(groups.values()) / fps))


def placeholder(clip):
    return replace(clip, first=(), last=(), samples=(), locked=False, gap=True, original=False, highlight=False)


def resized(movie, analysis, clip, start, end, cache):
    if (start, end) == (clip.start, clip.end):
        return clip
    if clip.highlight:
        fresh = fingerprint_highlight(movie, analysis, start, end, cache=cache, role=clip.role or "Highlight")
        return replace(fresh, muted=clip.muted, locked=clip.locked)
    return fingerprint_highlight(movie, analysis, start, end, cache=cache, candidate=clip)


def gap_candidate(movie, analysis, edits, index, cache, min_gap, section=None, manual=None):
    gap = edits[index]
    used = [c for c in edits if not c.gap]
    if manual is None:
        ranges = [*section.evidence, *((a, b, "") for a, b in section.alternatives)] if section else []
        if not ranges:
            ranges = [(gap.start, gap.end, "")]
        options = [c for c in analysis.candidates if not c.production_card and c.frames >= gap.frames
                   and not overlaps(c, gap) and (not section or any(a <= c.start < b or c.start <= a < c.end for a, b, _ in ranges))]
        options.sort(key=lambda c: (min(abs(c.start - a) for a, _, _ in ranges), -c.score))
    else:
        options = [manual]
    for candidate in options:
        if candidate.frames < gap.frames:
            continue
        if any(overlaps(candidate, c) for c in used):
            continue
        replacement = resized(movie, analysis, candidate, candidate.start, candidate.start + gap.frames, cache)
        trial = [*edits[:index], replacement, *edits[index + 1:]]
        try:
            validate(trial, movie.fps, analysis.frames, has_audio=movie.has_audio, min_gap=min_gap)
        except ValueError:
            continue
        return replacement
    raise ValueError("No valid full-length replacement was found. Choose another scene or leave the gap open.")


def cinema(movie, analysis, evidence, cache, progress, settings, existing=(), script=(), min_gap=5):
    if not movie.has_audio:
        return []
    available = [c for c in analysis.candidates if not c.production_card]
    starts = {c.start for c in available}
    by_start = {c.start: c for c in available}
    marked = list(existing)
    sections = [("Opening", settings["opening"], 0.18, script[0].evidence if script else ()),
                *[("Highlight", settings["length"], (i + 1) / (settings["count"] + 1), ())
                  for i in range(settings["count"])],
                ("Ending", settings["ending"], 0.82, script[-1].evidence if script else ())]
    for role, seconds, position, relevant in sections:
        if not seconds or settings.get(f"disable_{role}", False):
            continue
        desired = sum(kind == role and second > 0 for kind, second, _, _ in sections)
        if sum(c.role == role or role == "Highlight" and c.role not in {"Opening", "Ending"} for c in marked) >= desired:
            continue
        length = round(seconds * movie.fps)
        target = position * analysis.frames
        candidates = []
        for start in starts:
            end = start + length
            cursor = start
            while cursor < end and cursor in by_start:
                cursor = by_start[cursor].end
            if cursor < end or any(max(start, c.start) - min(end, c.end) < min_gap * movie.fps for c in marked):
                continue
            first, last = start / movie.fps, end / movie.fps
            if any(a < point < b for a, b in evidence.captions for point in (first, last)):
                continue
            sound = evidence.score(float(first + analysis.offset), float(last + analysis.offset))
            if sound <= .025:
                continue
            support = any(start <= a < end for a, _, _ in relevant)
            score = sound + .20 * by_start[start].score + float(support) - 2 * abs(start - target) / analysis.frames
            candidates.append((score, start))
        if candidates:
            _, start = max(candidates)
            progress(f"Preparing original audio · {role}")
            marked.append(fingerprint_highlight(movie, analysis, start, start + length, cache=cache, role=role))
    return marked


def bookends(clips):
    return ([c for c in clips if c.role == "Opening"] +
            [c for c in clips if c.role not in {"Opening", "Ending"}] +
            [c for c in clips if c.role == "Ending"])


def fit_recording(recording, clips, movie, analysis, cache, min_gap, progress, *, script=(), cues=()):
    """Fit picture to the recording's exact frame-rounded duration without editing its samples."""
    from .narration import section_blocks
    if any(c.gap for c in clips):
        raise ValueError("Fill unresolved picture gaps before synchronizing.")
    fps = movie.fps
    total = narration_frames(recording, fps)
    quiet = quiet_ranges(recording)
    aligned_blocks, sections, aligned = section_blocks(recording, clips, script, cues, quiet, fps)
    groups = []
    for clips_in_section, audio_start, audio_end in aligned_blocks:
        run = []
        partition = []
        for clip in clips_in_section:
            if clip.movie_audio:
                if run:
                    partition.append(run)
                    run = []
                partition.append([clip])
            else:
                run.append(clip)
        if run:
            partition.append(run)
        groups.append((partition, audio_start, audio_end))
    result, schedule = [], []
    picture_frame = bed_sample = audio_sample = narrated = 0
    for number, (partition, audio_start, audio_end) in enumerate(groups, 1):
        progress(f"Fitting recording · section {number}/{len(groups)}")
        runs = [run for run in partition if not run[0].movie_audio]
        section_end = total if audio_end == recording.frames else round(Fraction(audio_end, RATE) * fps)
        remaining = section_end - narrated
        available = sum(c.frames for run in runs for c in run)
        if available < remaining:
            raise ValueError(f"Section {number} needs {float(remaining / fps):.3f} s of narrated picture. "
                             f"{float(available / fps):.3f} s is selected. Add relevant footage or regenerate from cached analysis.")
        for run in partition:
            if run[0].movie_audio:
                result.extend(run)
                picture_frame += run[0].frames
                continue
            capacity = sum(c.frames for c in run)
            after = available - capacity
            minimum = max(1, remaining - after, sum(c.frames for c in run if c.locked or c.role))
            maximum = min(remaining - (1 if after else 0), capacity)
            if after:
                preferred = min(maximum, max(minimum, round(remaining * capacity / available)))
                floor = audio_sample + round(minimum * RATE / fps)
                ceiling = audio_sample + round(maximum * RATE / fps)
                pause = last_pause(quiet, floor, min(ceiling, audio_sample + round(preferred * RATE / fps)))
                if pause is None:
                    pause = last_pause(quiet, floor, ceiling)
                if pause is None:
                    raise ValueError(f"Section {number} needs a safe pause near an original-audio cut. Move the highlight or add relevant footage.")
                nearest = round(Fraction(pause - audio_sample, RATE) * fps)
                for size in (nearest, nearest - 1, nearest + 1):
                    if not minimum <= size <= maximum:
                        continue
                    physical = audio_sample + round((picture_frame + size) * RATE / fps) - round(picture_frame * RATE / fps)
                    if last_pause(quiet, physical, physical) is not None:
                        break
                else:
                    raise ValueError(f"No frame-aligned speech pause is available for section {number}. Move the original-audio cut.")
            else:
                size = remaining
            if not minimum <= size <= maximum:
                raise ValueError(f"Section {number} cannot fit the recording with its protected clips.")
            exact = fit_run(run, size, result[-1] if result else None, movie, analysis, cache, min_gap)
            result.extend(exact)
            count = round((picture_frame + size) * RATE / fps) - round(picture_frame * RATE / fps)
            end = min(audio_sample + count, recording.frames)
            if end > audio_sample:
                schedule.append((bed_sample, audio_sample, end))
            audio_sample, bed_sample = end, bed_sample + count
            picture_frame += size
            narrated += size
            remaining -= size
            available = after
    if audio_sample != recording.frames:
        raise ValueError("The timeline frame clock cannot cover the entire recording. Adjust an original-audio cut.")
    validate(result, fps, analysis.frames, has_audio=movie.has_audio, min_gap=min_gap)
    sections = [replace(section, evidence=tuple(
        (clip.start, clip.end, description) for a, b, description in section.evidence
        for clip in result if a <= clip.start < clip.end <= b and not clip.movie_audio
    )) for section in sections]
    return result, sections, tuple(schedule), bed_sample, aligned


def fit_run(run, frames, previous, movie, analysis, cache, min_gap):
    remaining = sum(c.frames for c in run) - frames
    edits = list(run)
    for index in range(len(edits) - 1, -1, -1):
        if not remaining:
            break
        clip = edits[index]
        if clip.locked or clip.role:
            continue
        if clip.frames > remaining:
            edits[index] = resized(movie, analysis, clip, clip.start, clip.end - remaining, cache)
            remaining = 0
            break
        left = edits[index - 1] if index else previous
        right = edits[index + 1] if index + 1 < len(edits) else None
        if left and right and adjacent_conflict(left, right, movie.fps, min_gap):
            continue
        edits.pop(index)
        remaining -= clip.frames
    if remaining:
        raise ValueError("Locked clips or protected cuts leave too much picture for this narration section.")
    return edits
