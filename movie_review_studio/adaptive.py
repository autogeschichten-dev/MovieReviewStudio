"""Narration-driven planning and frame-accurate picture edits."""
from dataclasses import replace
from bisect import bisect_left
from fractions import Fraction
from itertools import accumulate
from math import ceil

from .analysis import fingerprint_highlight
from .model import adjacent_conflict, overlaps, validate
from .narration import last_pause, quiet_ranges
from .render import RATE, audio_spans


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


def voiced_cues(quiet, cues):
    """Keep only physically quiet intervals outside timestamped speech."""
    speech = sorted((int(cue["start"] * RATE), ceil(cue["end"] * RATE)) for cue in cues)
    result = []
    index = 0
    for first, last in quiet:
        while index < len(speech) and speech[index][1] <= first:
            index += 1
        cursor = first
        for offset in range(index, len(speech)):
            start, end = speech[offset]
            if start >= last:
                break
            if start > cursor:
                result.append((cursor, start - 1))
            cursor = max(cursor, end + 1)
            if cursor > last:
                break
        if cursor <= last:
            result.append((cursor, last))
    return result


def join_removed_cut(clips, index, movie, analysis, cache):
    """Restore a source fragment split by an original-audio cut that has since been removed."""
    if not 0 < index < len(clips):
        return clips
    left, right = clips[index - 1:index + 1]
    if (left.end != right.start or left.movie_audio or right.movie_audio or left.locked or right.locked
            or left.role or right.role or left.gap or right.gap):
        return clips
    parent = next((clip for clip in getattr(analysis, "candidates", ())
                   if clip.start <= left.start < right.end <= clip.end and clip.shot == left.shot == right.shot), None)
    if parent:
        joined = resized(movie, analysis, parent, left.start, right.end, cache)
    elif left.shot is not None and left.shot == right.shot and (right.end - left.start) / movie.fps <= 5:
        # Both verified pieces came from one selected source range. The caller revalidates the joined cut.
        joined = replace(left, end=right.end, last=right.last, samples=left.samples + right.samples)
    else:
        return clips
    return [*clips[:index - 1], joined, *clips[index + 1:]]


def positions(picture, fps):
    cursor = samples = 0
    interruptions = []
    schedule = []
    for original, start, end in audio_spans(picture):
        count = round(end * RATE / fps) - round(start * RATE / fps)
        if original:
            if cursor and (not interruptions or samples != interruptions[-1]):
                interruptions.append(samples)
        else:
            schedule.append((samples, samples + count))
            samples += count
        cursor = end
    return schedule, samples, interruptions


def break_candidates(cuts, total, inserted, quiet, preferred, recording, fps):
    """Use the existing picture clock to project quiet intervals onto eligible source cuts."""
    candidates = set()
    samples_per_frame = Fraction(RATE) / fps
    part_starts = [start for start, _, _ in recording.boundaries[1:]]
    # Original audio inserts shift the sub-frame audio clock by less than one sample per edit.
    # The final trial checks the exact count using the same frame-clock math as the renderer.
    anchors = [preferred, *[round(Fraction(start, RATE) * fps) for start in part_starts]]
    radius = round(90 * fps)
    for low, high in quiet:
        begin = max(1, ceil(low / samples_per_frame) - 1)
        end = min(total - 1, int(high / samples_per_frame) + 1)
        if begin > end:
            continue
        if begin > preferred + radius or end < preferred - radius:
            if not any(low - RATE <= start <= high + RATE for start in part_starts):
                continue
        nearest = min(end, max(begin, preferred))
        candidates.update((begin, begin + 1, end, end - 1, nearest, nearest - 1, nearest + 1))
        first = bisect_left(cuts, begin)
        last = bisect_left(cuts, end + 1)
        candidates.update(cuts[first:last])
        for anchor in anchors[1:]:
            if begin <= anchor <= end:
                candidates.add(anchor)
    existing = set(cuts)
    previous = max(inserted, default=0)
    return sorted((frame for frame in candidates if previous < frame < total),
                  key=lambda frame: (abs(frame - preferred), frame not in existing, frame))


def insert_groups(body, opening, ending, groups, recording, quiet, movie, analysis, cache, min_gap, progress, notices):
    inserted = {}
    fps = movie.fps
    total = sum(c.frames for c in body)

    def candidate_picture(picture, offsets):
        out = list(opening)
        frame = 0
        for clip in picture:
            if frame in offsets:
                out.append(offsets[frame])
            out.append(clip)
            frame += clip.frames
        return [*out, *ending]

    for group, preferred in groups:
        offsets = list(accumulate(c.frames for c in body))
        candidates = break_candidates(offsets, total, inserted, quiet, preferred, recording, fps)
        if group.locked:
            candidates = [preferred] if preferred in candidates else []
        for frame in candidates:
            index = bisect_left(offsets, frame)
            trial = body
            if offsets[index] != frame:
                clip = body[index]
                if clip.locked or clip.role:
                    continue
                boundary = clip.end - (offsets[index] - frame)
                # Reject cheaply before decoding fresh fingerprints for the two sides.
                first = replace(clip, end=boundary)
                second = replace(clip, start=boundary)
                if adjacent_conflict(first, group, fps, min_gap) or adjacent_conflict(group, second, fps, min_gap):
                    continue
                try:
                    first = resized(movie, analysis, clip, clip.start, boundary, cache)
                    second = resized(movie, analysis, clip, boundary, clip.end, cache)
                except (OSError, RuntimeError, ValueError):
                    continue
                trial = [*body[:index], first, second, *body[index + 1:]]
            positions_by_frame = {**inserted, frame: group}
            proposed = candidate_picture(trial, positions_by_frame)
            _, capacity, boundaries = positions(proposed, fps)
            if capacity < recording.frames or any(
                    point < recording.frames and last_pause(quiet, point, point) is None for point in boundaries):
                continue
            try:
                validate(proposed, fps, analysis.frames, has_audio=movie.has_audio, min_gap=min_gap)
            except ValueError:
                continue
            body, inserted = trial, positions_by_frame
            if frame != preferred:
                note = f"Moved original-audio cut to {float(frame / fps):.1f} s of narration."
                notices.append(note)
                progress(note)
            break
        else:
            if group.locked:
                raise ValueError("A locked original-audio cut has no verified pause at its current position. Unlock it to allow recovery.")
            note = f"Skipped original audio at source {float(group.start / fps):.1f} s: no compatible verified pause. Undo restores this cut."
            notices.append(note)
            progress(note)
    return candidate_picture(body, inserted)


def grow_picture(body, originals, needed, movie, analysis, min_gap, script):
    """Extend only with unused cached footage that satisfies existing cut protections."""
    fps = movie.fps
    sources = [*body, *originals]
    permitted = [(a, b) for section in script for a, b, _ in section.evidence]
    permitted.extend(pair for section in script for pair in section.alternatives)
    def supported(clip):
        return not script or any(a <= clip.start < clip.end <= b for a, b in permitted)

    candidates = [c for c in getattr(analysis, "candidates", ()) if not c.production_card and supported(c)
                  and not any(overlaps(c, old) or (c.shot is not None and c.shot == old.shot) for old in sources)]
    result = list(body)
    for clip in sorted(candidates, key=lambda c: (-c.score, c.start)):
        if any(overlaps(clip, item) or (clip.shot is not None and clip.shot == item.shot) for item in result):
            continue
        placements = []
        for index in range(len(result) + 1):
            left = result[index - 1] if index else originals[0] if originals and originals[0].role == "Opening" else None
            right = result[index] if index < len(result) else originals[-1] if originals and originals[-1].role == "Ending" else None
            # Keep added picture between its neighbors in source order when both are narrated.
            if left and not left.movie_audio and left.start >= clip.start:
                continue
            if right and not right.movie_audio and clip.start >= right.start:
                continue
            if ((left and adjacent_conflict(left, clip, fps, min_gap))
                    or (right and adjacent_conflict(clip, right, fps, min_gap))):
                continue
            distance = min((abs(clip.start - neighbor.start) for neighbor in (left, right) if neighbor), default=0)
            placements.append((distance, index))
        if not placements:
            continue
        _, index = min(placements)
        result.insert(index, clip)
        needed -= clip.frames
        if needed <= 0:
            break
    return result


def fit_recording(recording, clips, movie, analysis, cache, min_gap, progress, *, script=(), cues=(), notices=None):
    """Fit picture first, then insert original audio only at verified narration pauses."""
    from .narration import section_blocks

    if any(c.gap for c in clips):
        raise ValueError("Fill unresolved picture gaps before synchronizing.")
    if notices is None:
        notices = []
    fps = movie.fps
    total = narration_frames(recording, fps)
    quiet = voiced_cues(quiet_ranges(recording), cues)
    opening = [c for c in clips if c.movie_audio and c.role == "Opening"]
    ending = [c for c in clips if c.movie_audio and c.role == "Ending"]
    body = []
    groups = []
    source_frames = 0
    join_at = None
    for clip in clips:
        if clip.movie_audio:
            if clip.role not in {"Opening", "Ending"}:
                groups.append((clip, source_frames))
                join_at = len(body)
            continue
        body.append(clip)
        if join_at is not None:
            body = join_removed_cut(body, join_at, movie, analysis, cache)
        join_at = None
        source_frames += clip.frames
    if source_frames < total:
        body = grow_picture(body, [*opening, *(group for group, _ in groups), *ending],
                            total - source_frames, movie, analysis, min_gap, script)
        expanded = sum(clip.frames for clip in body)
        if expanded < total:
            raise ValueError(f"Narration needs {float(total / fps):.3f} s of picture. "
                             f"{float(expanded / fps):.3f} s could be placed using the selected picture and eligible cached cuts.")
        note = f"Added {float((expanded - source_frames) / fps):.1f} s from cached footage to fit narration."
        notices.append(note)
        progress(note)

    def fit_sections(blocks):
        picture = []
        narrated = 0
        for number, (items, _, audio_end) in enumerate(blocks, 1):
            progress(f"Fitting narrated picture · section {number}/{len(blocks)}")
            finish = total if audio_end == recording.frames else round(Fraction(audio_end, RATE) * fps)
            amount = finish - narrated
            if sum(clip.frames for clip in items) < amount:
                raise ValueError("This script section does not have enough picture for its recorded duration.")
            picture.extend(fit_run(items, amount, picture[-1] if picture else None, movie, analysis, cache, min_gap))
            narrated = finish
        if narrated != total:
            raise ValueError("The fitted picture does not cover the recording.")
        return picture

    sections, aligned = list(script), False
    if script and cues:
        try:
            blocks, aligned_sections, aligned = section_blocks(recording, body, script, cues, quiet, fps)
            body_fitted = fit_sections(blocks)
            sections = aligned_sections
        except (ValueError, RuntimeError):
            aligned = False
            note = "Could not apply precise script timing. Preserved the existing order of narrated picture."
            notices.append(note)
            progress(note)
    if not aligned:
        body_fitted = fit_sections([(body, 0, recording.frames)])
    groups = [(group, frame if group.locked else round(Fraction(frame * total, max(1, source_frames)))) for group, frame in groups]
    result = insert_groups(body_fitted, opening, ending, groups, recording, quiet, movie, analysis,
                           cache, min_gap, progress, notices)
    spans, length, _ = positions(result, fps)
    if length < recording.frames:
        raise ValueError("The fitted frame clock leaves an uncovered narration sample.")
    schedule = tuple((first, first, min(end, recording.frames)) for first, end in spans if first < recording.frames)
    validate(result, fps, analysis.frames, has_audio=movie.has_audio, min_gap=min_gap)
    sections = [replace(section, evidence=tuple(
        (clip.start, clip.end, description) for clip in result if not clip.movie_audio
        for a, b, description in section.evidence if a <= clip.start < clip.end <= b
    )) for section in sections]
    return result, sections, schedule, length, aligned


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
