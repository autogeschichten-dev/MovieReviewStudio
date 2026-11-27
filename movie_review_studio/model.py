"""Frame-indexed edit decisions and montage validation."""
from dataclasses import dataclass, replace
from fractions import Fraction
from math import ceil
from typing import Protocol

import numpy as np

MAX_CLIP_SECONDS = 5
DEFAULT_SOURCE_GAP = 5
FIXED_MIN_SECONDS = 960
FIXED_GENERATION_MIN_SECONDS = 1020
FIXED_MAX_SECONDS = 1080
NARRATION_SAMPLE_RATE = 48000


def narration_frames(samples, fps):
    return ceil(Fraction(samples, NARRATION_SAMPLE_RATE) * fps)


def adaptive_frames(clips, fps, samples):
    return narration_frames(samples, fps) + sum(clip.frames for clip in clips if clip.movie_audio)


def validate_duration(clips, fps, *, narration_samples=None):
    frames = sum(clip.frames for clip in clips)
    if narration_samples is not None:
        expected = adaptive_frames(clips, fps, narration_samples)
        if frames != expected:
            raise ValueError("The picture duration does not match the recording and enabled original audio. Run Merge & Synchronize.")
    elif not FIXED_MIN_SECONDS <= frames / fps <= FIXED_MAX_SECONDS:
        raise ValueError("The final montage must be 16–18 minutes including highlights.")


def has_picture_gap(clips):
    return any(clip.gap or any(part.gap for part in clip.picture) for clip in clips)



@dataclass(frozen=True)
class Signature:
    phash: int
    pixels: str  # 16×16 grayscale thumbnail, hex encoded


@dataclass(frozen=True)
class Clip:
    start: int
    end: int
    score: float
    first: tuple[Signature, ...]
    last: tuple[Signature, ...]
    samples: tuple[Signature, ...]
    highlight: bool = False
    shot: int | None = None
    cuts: tuple[int, ...] = ()  # Legacy project boundaries, retained for repair diagnostics only.

    original: bool = False
    production_card: bool = False
    muted: bool = False
    role: str = ""
    locked: bool = False
    gap: bool = False
    picture: tuple["Clip", ...] = ()  # Original-audio groups own independent picture edits.
    legacy_shots: tuple[int, ...] = ()

    @property
    def movie_audio(self):
        return (self.highlight and not self.muted) or self.original

    @property
    def frames(self):
        return self.end - self.start


def highlight_cuts(start: int, end: int, fps: Fraction, shot_starts=()) -> tuple[int, ...]:
    """Split a continuous source range at shot cuts and balance any long pieces."""
    maximum = int(MAX_CLIP_SECONDS * fps)
    boundaries = [start]
    for frame in sorted(set(shot_starts)):
        if boundaries[-1] + 2 <= frame <= end - 2:
            boundaries.append(frame)
    boundaries.append(end)
    cuts = []
    for left, right in zip(boundaries, boundaries[1:]):
        length = right - left
        pieces = ceil(Fraction(length, maximum))
        cuts.extend(left + length * piece // pieces for piece in range(1, pieces))
        cuts.append(right)
    return tuple(cuts[:-1])


def video_segments(clip: Clip, fps: Fraction) -> tuple[Clip, ...]:
    if not clip.highlight:
        return (clip,)
    if clip.picture:
        return tuple(replace(part, highlight=True, muted=clip.muted, original=clip.original,
                             role=clip.role, locked=clip.locked) for part in clip.picture)
    cuts = clip.cuts or highlight_cuts(clip.start, clip.end, fps)
    bounds = (clip.start, *cuts, clip.end)
    return tuple(replace(clip, start=start, end=end, cuts=(),
                         shot=clip.legacy_shots[i] if i < len(clip.legacy_shots) else clip.shot)
                 for i, (start, end) in enumerate(zip(bounds, bounds[1:])))


@dataclass(frozen=True)
class TimelineEntry:
    edit_index: int
    segment: Clip
    position: int


def timeline_entries(clips: list[Clip], fps: Fraction) -> list[TimelineEntry]:
    entries = []
    cursor = 0
    for index, clip in enumerate(clips):
        for segment in video_segments(clip, fps):
            entries.append(TimelineEntry(index, segment, cursor))
            cursor += segment.frames
    return entries


class HighlightRecommender(Protocol):
    def recommend(self, candidates: list[Clip], fps: Fraction) -> list[tuple[int, int]]:
        """Return proposed source frame ranges for manual review."""


class ClipRanker(Protocol):
    def __call__(self, clip: Clip) -> float:
        """Rank a candidate. Implementations may use future narration context."""


def similar(left: Signature, right: Signature) -> bool:
    if (left.phash ^ right.phash).bit_count() > 12:
        return False
    a = np.frombuffer(bytes.fromhex(left.pixels), dtype=np.uint8).astype(np.int16)
    b = np.frombuffer(bytes.fromhex(right.pixels), dtype=np.uint8)
    return float(np.mean(np.abs(a - b))) <= 19


def source_gap(left: Clip, right: Clip) -> int:
    """Space between disjoint ranges in either source order. Negative means overlap."""
    return max(right.start - left.end, left.start - right.end)


def cut_issue(left: Clip, right: Clip, fps=None, min_gap=DEFAULT_SOURCE_GAP):
    if left.picture:
        left = left.picture[-1]
    if right.picture:
        right = right.picture[0]
    if left.gap or right.gap:
        return None
    if left.shot is not None and left.shot == right.shot:
        return "continuous_source"
    if any(similar(a, b) for a in left.last for b in right.first) or any(
        similar(a, b) for a in left.samples for b in right.samples
    ):
        return "similar_visuals"
    if fps is not None:
        gap = source_gap(left, right)
        if gap < 0:
            return "overlap"
        if gap == 0:
            return "contiguous_source"
        if gap / fps < max(DEFAULT_SOURCE_GAP, min_gap):
            return "insufficient_source_gap"
    return None


def adjacent_conflict(left: Clip, right: Clip, fps=None, min_gap=DEFAULT_SOURCE_GAP) -> bool:
    return cut_issue(left, right, fps, min_gap) is not None


CUT_MESSAGES = {
    "continuous_source": "look too similar at this cut (same continuous source shot)",
    "similar_visuals": "look too similar at this cut",
    "overlap": "overlap in the source",
    "contiguous_source": "touch in the source and cannot be consecutive",
    "insufficient_source_gap": "are closer together in the source than the configured separation",
}


def overlaps(left: Clip, right: Clip) -> bool:
    return any(a.start < b.end and b.start < a.end
               for a in left.picture or (left,) for b in right.picture or (right,))


def fragment_issues(clips, fps, source_frames, min_gap=DEFAULT_SOURCE_GAP):
    """Audit actual rendered picture intervals, independent of audio grouping."""
    entries = list(enumerate(timeline_entries(clips, fps), 1))
    issues = []

    def detail(number, entry):
        clip = entry.segment
        return {"fragment_id": f"v{number}", "edit": entry.edit_index + 1, "source_shot_id": clip.shot,
                "source_start_frame": clip.start, "source_end_frame": clip.end,
                "source_start_seconds": float(clip.start / fps), "source_end_seconds": float(clip.end / fps),
                "audio_mode": "original" if clip.movie_audio else "narration"}

    def issue(reason, left, right=None):
        issues.append({"reason": reason, "left": detail(*left), "right": detail(*right) if right else None,
                       "source_gap_seconds": float(source_gap(left[1].segment, right[1].segment) / fps) if right else None,
                       "replacement_result": "unresolved"})

    visible = []
    for row in entries:
        _, entry = row
        clip = entry.segment
        if clip.gap:
            continue
        if not 0 <= clip.start < clip.end <= source_frames:
            issue("invalid_source_range", row)
        if clip.frames / fps > MAX_CLIP_SECONDS:
            issue("fragment_too_long", row)
        visible.append(row)
    for left, right in zip(entries, entries[1:]):
        a, b = left[1].segment, right[1].segment
        if a.gap or b.gap:
            continue
        gap = source_gap(a, b)
        if gap == 0:
            issue("contiguous_source", left, right)
        elif 0 < gap / fps < max(DEFAULT_SOURCE_GAP, min_gap):
            issue("insufficient_source_gap", left, right)
        reason = cut_issue(a, b)
        if reason:
            issue(reason, left, right)
    ordered = sorted(visible, key=lambda row: row[1].segment.start)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            if right[1].segment.start >= left[1].segment.end:
                break
            if overlaps(left[1].segment, right[1].segment):
                issue("overlap", left, right)
    return issues


def validate(clips: list[Clip], fps: Fraction, source_frames: int, *,
             final: bool = False, has_audio: bool = True, min_gap: int = DEFAULT_SOURCE_GAP, narration_samples=None) -> None:
    if min_gap < 0:
        raise ValueError("Minimum source separation cannot be negative.")
    if not clips:
        raise ValueError("The timeline is empty.")
    if final and has_picture_gap(clips):
        raise ValueError("Fill unresolved picture gaps before export.")
    highlights = 0
    roles = set()
    for index, clip in enumerate(clips):
        number = index + 1
        if not 0 <= clip.start < clip.end <= source_frames:
            raise ValueError(f"Clip {number} lies outside the source.")
        duration = clip.frames / fps
        if clip.movie_audio and not has_audio:
            raise ValueError(f"Clip {number} needs movie audio.")
        if clip.gap and final:
            raise ValueError("Fill unresolved picture gaps before export.")
        if clip.highlight:
            cinematic = clip.role in {"Opening", "Ending"}
            if cinematic:
                if clip.role in roles or index != (0 if clip.role == "Opening" else len(clips) - 1):
                    raise ValueError(f"The {clip.role.lower()} must stay at its end of the timeline.")
                roles.add(clip.role)
            else:
                highlights += 1
            low, high = (1, 30) if cinematic else (1, 15) if clip.role == "Highlight" else (10, 15)
            if not has_audio or not low <= duration <= high:
                raise ValueError(f"Highlight {number} needs movie audio and a length of {low}–{high} seconds.")
            if clip.picture and (any(part.highlight or part.picture or (not part.gap and part.shot is None) for part in clip.picture)
                                 or sum(part.frames for part in clip.picture) != clip.frames):
                raise ValueError(f"Original-audio group {number} needs identified picture fragments matching its audio duration.")
    issues = fragment_issues(clips, fps, source_frames, min_gap)
    if issues:
        first = issues[0]
        left, right = first["left"], first["right"]
        reason = first["reason"]
        if right:
            raise ValueError(f"Clips {left['fragment_id']} and {right['fragment_id']} {CUT_MESSAGES[reason]}.")
        if reason == "fragment_too_long":
            raise ValueError(f"Fragment {left['fragment_id']} exceeds 5 seconds.")
        raise ValueError(f"Fragment {left['fragment_id']} lies outside the source.")
    if highlights > 3:
        raise ValueError("The montage supports up to three original-audio highlight groups.")
    if final:
        validate_duration(clips, fps, narration_samples=narration_samples)


def timeline_validation_report(clips, fps, source_frames, *, has_audio=True, min_gap=DEFAULT_SOURCE_GAP,
                               fitted_duration=False, narration_samples=None):
    entries = timeline_entries(clips, fps)
    issues = fragment_issues(clips, fps, source_frames, min_gap)
    try:
        if has_picture_gap(clips):
            raise ValueError("Fill unresolved picture gaps before export.")
        validate(clips, fps, source_frames, final=narration_samples is not None or not fitted_duration,
                 has_audio=has_audio, min_gap=min_gap, narration_samples=narration_samples)
    except ValueError as error:
        failure = str(error)
    else:
        failure = None
    overlapping = [[row["left"]["edit"], row["right"]["edit"]] for row in issues if row["reason"] == "overlap"]
    return {"passed": failure is None, "overlapping_edits": overlapping, "first_error": failure,
            "source_fps": str(fps), "minimum_source_gap_seconds": max(DEFAULT_SOURCE_GAP, min_gap),
            "maximum_fragment_seconds": MAX_CLIP_SECONDS, "duration_fitted_to_narration": fitted_duration or narration_samples is not None,
            "duration_seconds": float(sum(clip.frames for clip in clips) / fps), "violations": issues,
            "adjacent_source_fragment_violations": len({(row["left"]["fragment_id"], row["right"]["fragment_id"])
                                                       for row in issues if row["reason"] in CUT_MESSAGES
                                                       and row["right"] and abs(int(row["left"]["fragment_id"][1:])
                                                       - int(row["right"]["fragment_id"][1:])) == 1}),
            "source_overlap_violations": len(overlapping),
            "touching_source_interval_violations": sum(row["reason"] == "contiguous_source" for row in issues),
            "fragments_longer_than_five_seconds": sum(row["reason"] == "fragment_too_long" for row in issues),
            "original_audio_visual_exceptions": 0,
            "fragments": [{"fragment_id": f"v{i}", "edit": entry.edit_index + 1, "source_shot_id": entry.segment.shot,
                           "source_start_frame": entry.segment.start, "source_end_frame": entry.segment.end,
                           "timeline_start_frame": entry.position, "audio_mode": "original" if entry.segment.movie_audio else "narration",
                           "duration_seconds": float(entry.segment.frames / fps)} for i, entry in enumerate(entries, 1)]}


def choose(candidates: list[Clip], highlights: list[Clip], fps: Fraction,
           source_frames: int, target_seconds: int, rank: ClipRanker | None = None, *, min_gap=DEFAULT_SOURCE_GAP) -> list[Clip]:
    if not 2 <= len(highlights) <= 3 or not FIXED_MIN_SECONDS <= target_seconds <= FIXED_MAX_SECONDS:
        raise ValueError("Select 2–3 highlights and a target of 16–18 minutes.")
    rank = rank or (lambda clip: clip.score)
    lanes: list[list[Clip]] = [[] for _ in range(12)]
    for clip in sorted(candidates, key=rank, reverse=True):
        if not any(overlaps(clip, highlight) for highlight in highlights):
            lanes[min(11, clip.start * 12 // source_frames)].append(clip)
    result: list[Clip] = []
    remaining = sorted(highlights, key=lambda clip: clip.start)
    total = 0
    target = min(int(FIXED_MAX_SECONDS * fps), ceil(target_seconds * fps))
    maximum = int(FIXED_MAX_SECONDS * fps)
    lane_number = 0
    while total < target or remaining:
        previous = result[-1] if result else None
        allowance = maximum - total - sum(clip.frames for clip in remaining)
        highlight_due = remaining and total >= target * (len(highlights) - len(remaining) + 1) / (len(highlights) + 1)
        if highlight_due and (previous is None or not adjacent_conflict(previous, remaining[0], fps, min_gap)):
            selected = remaining.pop(0)
        else:
            selected = None
            for offset in range(len(lanes)):
                lane_index = (lane_number + offset) % len(lanes)
                lane = lanes[lane_index]
                for index, candidate in enumerate(lane):
                    if candidate.frames > allowance:
                        continue
                    if previous and adjacent_conflict(previous, candidate, fps, min_gap):
                        continue
                    if any(overlaps(used, candidate) for used in result):
                        continue
                    if remaining and adjacent_conflict(candidate, remaining[0], fps, min_gap):
                        continue
                    selected = lane.pop(index)
                    lane_number = lane_index + 1
                    break
                if selected:
                    break
            if not selected:
                if remaining and (previous is None or not adjacent_conflict(previous, remaining[0], fps, min_gap)):
                    selected = remaining.pop(0)
                else:
                    break
        result.append(selected)
        total += selected.frames
    if remaining or total < ceil(FIXED_MIN_SECONDS * fps):
        raise ValueError("There is insufficient distinct usable footage to build a valid 16-minute montage.")
    validate(result, fps, source_frames, final=True, min_gap=min_gap)
    return result


def replacement(clips: list[Clip], index: int, candidates: list[Clip], *, fps=None, min_gap=DEFAULT_SOURCE_GAP) -> Clip:
    old = clips[index]
    used = clips[:index] + clips[index + 1:]
    left = clips[max(0, index - 1):index]
    right = clips[index + 1:index + 2]
    for clip in sorted(candidates, key=lambda item: (abs(item.frames - old.frames), -item.score)):
        if overlaps(clip, old) or any(overlaps(clip, item) for item in used):
            continue
        if not any(adjacent_conflict(neighbor, clip, fps, min_gap) for neighbor in left) and not any(
            adjacent_conflict(clip, neighbor, fps, min_gap) for neighbor in right
        ):
            return clip
    raise ValueError("No conflict-free replacement was found.")
