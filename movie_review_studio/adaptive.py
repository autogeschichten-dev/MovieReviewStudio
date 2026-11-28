"""Narration-driven planning and frame-accurate picture edits."""
from collections import Counter
from dataclasses import replace
from bisect import bisect_left
from fractions import Fraction
from itertools import accumulate
from pathlib import Path
from math import ceil

from .analysis import fingerprint_highlight
from .model import adaptive_frames, adjacent_conflict, has_picture_gap, narration_frames as sample_frames, overlaps, validate
from .narration import last_pause, quiet_ranges
from .render import RATE, audio_spans


def narration_frames(recording, fps):
    return sample_frames(recording.frames, fps)


def duration_totals(recording, clips, fps):
    groups = {"Opening": 0, "Highlight": 0, "Ending": 0}
    for clip in clips:
        if clip.movie_audio:
            role = clip.role if clip.role in ("Opening", "Ending") else "Highlight"
            groups[role] += clip.frames
    narrated = recording.duration if recording else 0
    return (narrated, *(float(frames / fps) for frames in groups.values()),
            float(adaptive_frames(clips, fps, recording.frames if recording else 0) / fps))


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


def replace_audio_picture(movie, analysis, cache, edits, index, fragment, min_gap, manual=None):
    import cv2
    from .analysis import cutaway_without_visible_speaker, source_fragment

    group = edits[index]
    old = group.picture[fragment]
    audio_start = group.start + sum(c.frames for c in group.picture[:fragment])
    context = round(45 * movie.fps)
    candidates = [manual] if manual else sorted(analysis.candidates, key=lambda c: abs(c.start - audio_start))
    detector = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    for candidate in candidates:
        if candidate.production_card or candidate.frames < old.frames or candidate.shot is None:
            continue
        in_sync = candidate.start <= audio_start and audio_start + old.frames <= candidate.end
        start = audio_start if in_sync else candidate.start
        if not in_sync:
            if abs(start - audio_start) > context or not cutaway_without_visible_speaker(
                    replace(movie, offset=analysis.offset), candidate, detector):
                continue
        piece = source_fragment(movie, analysis, start, start + old.frames, cache)
        if any(piece.shot == c.shot for i, c in enumerate(group.picture) if i != fragment and not c.gap):
            continue
        picture = [*group.picture[:fragment], piece, *group.picture[fragment + 1:]]
        changed = replace(group, picture=tuple(picture))
        try:
            validate([*edits[:index], changed, *edits[index + 1:]], movie.fps, analysis.frames,
                     has_audio=movie.has_audio, min_gap=min_gap)
        except ValueError:
            continue
        return changed
    raise ValueError("No replacement can preserve this original-audio passage with the required picture separation.")


def audio_structure(settings, clips):
    if settings.get("omit_auto_audio"):
        return {}
    expected = {"Opening": int(bool(settings["opening"]) and not settings.get("disable_Opening")),
                "Highlight": settings["count"],
                "Ending": int(bool(settings["ending"]) and not settings.get("disable_Ending"))}
    present = dict.fromkeys(expected, 0)
    for group in clips:
        if group.highlight and group.movie_audio:
            role = group.role if group.role in {"Opening", "Ending"} else "Highlight"
            present[role] += 1
    return {role: [expected[role], present[role]] for role in expected if present[role] < expected[role]}


def cinema(movie, analysis, evidence, cache, progress, settings, existing=(), script=(), min_gap=5, *,
           reserved=(), omissions=None, bookend_neighbors=(), audit=None, pause_fractions=()):
    if omissions is None:
        omissions = []
    if audit is None:
        audit = []
    expected = {role: count for role, count in (
        ("Opening", int(bool(settings["opening"]))), ("Highlight", settings["count"]),
        ("Ending", int(bool(settings["ending"])))) if count and not settings.get(f"disable_{role}")}
    if not movie.has_audio:
        if expected:
            omissions.append("The movie has no original audio stream.")
        return []
    screened = {}
    def joins(group):
        if not bookend_neighbors:
            return True
        if group.role == "Opening":
            return not adjacent_conflict(group, bookend_neighbors[0], movie.fps, min_gap)
        if group.role == "Ending":
            return not adjacent_conflict(bookend_neighbors[-1], group, movie.fps, min_gap)
        return True

    available = [c for c in analysis.candidates if not c.production_card]
    starts = {c.start for c in available}
    by_start = {c.start: c for c in available}
    marked = []
    for clip in existing:
        try:
            validate([clip], movie.fps, analysis.frames, has_audio=movie.has_audio, min_gap=min_gap)
        except ValueError:
            if clip.locked:
                raise ValueError("Unlock the legacy original-audio group before rebuilding its picture.")
            try:
                repaired = fingerprint_highlight(movie, analysis, clip.start, clip.end, cache=cache, role=clip.role,
                                                 reserved=(*marked, *reserved), min_gap=min_gap, screened=screened)
            except ValueError:
                progress("Skipped legacy audio group: no suitable non-adjacent picture.")
            else:
                marked.append(replace(repaired, muted=clip.muted))
        else:
            marked.append(clip)
    sections = [("Opening", settings["opening"], 0.18, script[0].evidence if script else ()),
                *[("Highlight", settings["length"], (i + 1) / (settings["count"] + 1), ())
                  for i in range(settings["count"])],
                ("Ending", settings["ending"], 0.82, script[-1].evidence if script else ())]
    rows = []
    seen = Counter()
    missing = {role: wanted - found for role, (wanted, found) in audio_structure(settings, marked).items()}
    for role, seconds, position, relevant in sections:
        seen[role] += 1
        if seen[role] > missing.get(role, 0):
            continue
        if role == "Highlight" and pause_fractions:
            position = min(pause_fractions, key=lambda value: abs(value - position))
        related = [(a, b) for a, b, _ in relevant]
        counts = Counter()
        row = {"role": role, "slot": seen[role], "observed_dialogue_cues": len(evidence.captions),
               "requested_seconds": seconds, "counts": counts}
        audit.append(row)
        attempts = []
        durations = (seconds,) if role != "Highlight" else tuple(dict.fromkeys((seconds, 12, 10, 15)))
        for duration in durations:
            length = round(duration * movie.fps)
            for start in starts:
                end = start + length
                if end > analysis.frames:
                    counts["passage_outside_movie"] += 1
                    continue
                first, last = start / movie.fps, end / movie.fps
                boundaries = sum(any(a < point < b for a, b in evidence.captions) for point in (first, last))
                left = max(0, int((first + analysis.offset) * 20))
                right = min(len(evidence.peaks), int((last + analysis.offset) * 20))
                peaks = evidence.peaks[left:right]
                if not peaks or max(peaks) < .003:
                    counts["no_audible_waveform"] += 1
                    continue
                sound = evidence.score(float(first + analysis.offset), float(last + analysis.offset))
                support = any(start <= a < end for a, _, _ in relevant)
                score = (sound + .20 * by_start[start].score + float(support) - boundaries
                         - 2 * abs(start / analysis.frames - position) - .04 * abs(duration - seconds))
                attempts.append((score, start, end))
        bins = {}
        for _, start, end in sorted(attempts, reverse=True):
            buckets = bins.setdefault(start // max(1, round(30 * movie.fps)), [])
            if len(buckets) < 8:
                buckets.append((start, end))
        diverse = [(start, end) for i in range(8) for bucket in bins.values() if i < len(bucket)
                   for start, end in [bucket[i]]]
        rows.append((role, related, diverse, row))
    best = list(marked)
    branches = 0

    def reserve(index, chosen):
        nonlocal best, branches
        if len(chosen) > len(best):
            best = list(chosen)
        if index == len(rows):
            return len(chosen) == len(marked) + len(rows)
        role, related, candidates, row = rows[index]
        counts = row["counts"]
        for start, end in candidates[:96]:
            if branches >= 256:
                counts["allocation_search_budget"] += 1
                break
            if any(start < c.end and c.start < end for c in chosen if c.movie_audio):
                counts["reserved_audio"] += 1
                continue
            branches += 1
            counts["passages_tried"] += 1
            try:
                group = fingerprint_highlight(movie, analysis, start, end, cache=cache, role=role,
                    reserved=(*chosen, *reserved), min_gap=min_gap, screened=screened,
                    rejected=counts, related=related)
            except ValueError:
                counts["no_complete_picture_chain"] += 1
                continue
            if not joins(group):
                counts["bookend_join"] += 1
                continue
            counts["valid_passages"] += 1
            if reserve(index + 1, [*chosen, group]):
                row["selected_source_frames"] = [start, end]
                return True
            counts["backtracked"] += 1
        if len(chosen) + len(rows) - index - 1 > len(best):
            reserve(index + 1, chosen)
        return False

    progress("Reserving original-audio picture before narrated footage…")
    reserve(0, marked)
    marked = best
    for role, (count, found) in audio_structure(settings, marked).items():
        omissions.append(f"{role}: found {found} of {count} requested original-audio sections with valid picture; see audio allocation report.")
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


def break_candidates(cuts, total, inserted, quiet, preferred, fps):
    """Use the existing picture clock to project quiet intervals onto eligible source cuts."""
    candidates = set()
    samples_per_frame = Fraction(RATE) / fps
    # Original audio inserts shift the sub-frame audio clock by less than one sample per edit.
    # The final trial checks the exact count using the same frame-clock math as the renderer.
    for low, high in quiet:
        begin = max(1, ceil(low / samples_per_frame) - 1)
        end = min(total - 1, int(high / samples_per_frame) + 1)
        if begin > end:
            continue
        first = bisect_left(cuts, begin)
        last = bisect_left(cuts, end + 1)
        candidates.update(cuts[first:last])
        sample = min(high, max(low, round(preferred * samples_per_frame)))
        frame = round(sample / samples_per_frame)
        candidates.update((frame - 1, frame, frame + 1))
    previous = max(inserted, default=0)
    return sorted((frame for frame in candidates if previous < frame < total),
                  key=lambda frame: (abs(frame - preferred), frame))


def retime_cut(body, offsets, frame, movie, analysis, cache):
    index = bisect_left(offsets, frame)
    for cut in (index, index - 1):
        if not 0 <= cut < len(body) - 1:
            continue
        left, right = body[cut:cut + 2]
        shift = frame - offsets[cut]
        sizes = left.frames + shift, right.frames - shift
        if (left.locked or right.locked or left.role or right.role or left.shot == right.shot
                or not all(0 < size / movie.fps <= 5 for size in sizes)):
            continue
        edited = []
        for clip, size in zip((left, right), sizes):
            parent = next((c for c in getattr(analysis, "candidates", ())
                           if c.shot == clip.shot and c.start <= clip.start and clip.start + size <= c.end), None)
            if size <= clip.frames:
                parent = clip
            if parent is None:
                break
            edited.append(resized(movie, analysis, parent, clip.start, clip.start + size, cache))
        if len(edited) == 2:
            return [*body[:cut], *edited, *body[cut + 2:]]
    return None


def insert_groups(body, opening, ending, groups, recording, quiet, movie, analysis, cache, min_gap, progress, notices, omissions):
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

    best = None
    attempts = 0

    def search(number, picture, offsets, moved, skipped):
        nonlocal best, attempts
        if best and len(skipped) >= len(best[2]):
            return False
        if number == len(groups):
            result = candidate_picture(picture, offsets)
            try:
                validate(result, fps, analysis.frames, has_audio=movie.has_audio, min_gap=min_gap)
            except ValueError:
                return False
            best = result, moved, skipped
            return not skipped
        group, preferred = groups[number]
        cuts = list(accumulate(c.frames for c in picture))
        candidates = break_candidates(cuts, total, offsets, quiet, preferred, fps)
        if group.locked:
            candidates = [preferred] if preferred in candidates else []
        for frame in candidates:
            if attempts >= 320:
                break
            attempts += 1
            at = bisect_left(cuts, frame)
            try:
                trial = picture if cuts[at] == frame else retime_cut(picture, cuts, frame, movie, analysis, cache)
            except (OSError, RuntimeError, ValueError):
                continue
            if trial is None:
                continue
            anchors = {**offsets, frame: group}
            proposed = candidate_picture(trial, anchors)
            if any(audio not in proposed for audio in anchors.values()):
                continue
            _, capacity, boundaries = positions(proposed, fps)
            if capacity < recording.frames or any(
                    point < recording.frames and last_pause(quiet, point, point) is None for point in boundaries):
                continue
            try:
                validate(proposed, fps, analysis.frames, has_audio=movie.has_audio, min_gap=min_gap)
            except ValueError:
                continue
            next_moved = moved if frame == preferred else [*moved, frame]
            if search(number + 1, trial, anchors, next_moved, skipped):
                return True
        if group.locked:
            return False
        return search(number + 1, picture, offsets, moved, [*skipped, group])

    search(0, body, {}, [], [])
    if best is None:
        validate(candidate_picture(body, {}), fps, analysis.frames, has_audio=movie.has_audio, min_gap=min_gap)
        raise ValueError("A protected original-audio cut could not be aligned to a verified pause.")
    result, moved, skipped = best
    for frame in moved:
        note = f"Moved original-audio cut to {float(frame / fps):.1f} s of narration."
        notices.append(note)
        progress(note)
    for group in skipped:
        note = f"Skipped original audio at source {float(group.start / fps):.1f} s: no compatible verified pause and strict picture join."
        notices.append(note)
        if omissions is not None:
            omissions.append(note)
        progress(note)
    return result


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


def fit_recording(recording, clips, movie, analysis, cache, min_gap, progress, *, script=(), cues=(), notices=None, omissions=None, restored=()):
    """Fit picture first, then insert original audio only at verified narration pauses."""
    from .narration import section_blocks

    if has_picture_gap(clips):
        raise ValueError("Fill unresolved picture gaps before synchronizing.")
    if notices is None:
        notices = []
    fps = movie.fps
    total = narration_frames(recording, fps)
    quiet = voiced_cues(quiet_ranges(recording), cues)
    clips = [*clips, *restored]
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
    original = [*opening, *(group for group, _ in groups), *ending]
    conflicts = [c for c in body if any(overlaps(c, group) for group in original)]
    free = [c for c in body if c not in conflicts]
    while opening and free and adjacent_conflict(opening[-1], free[0], fps, min_gap):
        conflicts.append(free.pop(0))
    while ending and free and adjacent_conflict(free[-1], ending[0], fps, min_gap):
        conflicts.append(free.pop())
    if any(c.locked for c in conflicts):
        raise ValueError("Reserved original audio overlaps protected narrated picture.")
    if conflicts:
        body = [c for c in body if c not in conflicts]
        note = f"Reallocated {len(conflicts)} narrated clips to reserve original-audio picture."
        notices.append(note)
        progress(note)
    available = sum(clip.frames for clip in body)
    if available < total:
        body = grow_picture(body, original, total - available, movie, analysis, min_gap, script)
        expanded = sum(clip.frames for clip in body)
        if expanded < total:
            raise ValueError(f"Narration needs {float(total / fps):.3f} s of picture. "
                             f"{float(expanded / fps):.3f} s could be placed using the selected picture and eligible cached cuts.")
        note = f"Added {float((expanded - available) / fps):.1f} s from cached footage to fit narration."
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
    newly_restored = [group for group in restored if group.role not in {"Opening", "Ending"}]
    anchors = {group: round(total * Fraction(i + 1, len(newly_restored) + 1))
               for i, group in enumerate(newly_restored)}
    groups = [(group, anchors.get(group, frame if group.locked else round(Fraction(frame * total, max(1, source_frames)))))
              for group, frame in groups]
    if anchors:
        groups.sort(key=lambda item: item[1])
    result = insert_groups(body_fitted, opening, ending, groups, recording, quiet, movie, analysis,
                           cache, min_gap, progress, notices, omissions)
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
