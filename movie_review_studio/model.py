"""Frame-indexed edit decisions and montage validation."""
from dataclasses import dataclass, replace
from fractions import Fraction
from math import ceil
from typing import Protocol

import numpy as np

MAX_CLIP_SECONDS = 5
DEFAULT_SOURCE_GAP = 5


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
    cuts: tuple[int, ...] = ()  # Internal video boundaries of a continuous highlight.

    original: bool = False
    production_card: bool = False
    muted: bool = False
    role: str = ""

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
    cuts = clip.cuts or highlight_cuts(clip.start, clip.end, fps)
    bounds = (clip.start, *cuts, clip.end)
    return tuple(replace(clip, start=start, end=end, cuts=()) for start, end in zip(bounds, bounds[1:]))


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
        if gap / fps < min_gap:
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
    return left.start < right.end and right.start < left.end


def validate(clips: list[Clip], fps: Fraction, source_frames: int, *,
             final: bool = False, has_audio: bool = True,
             min_gap: int = DEFAULT_SOURCE_GAP, enforce_separation: bool = True) -> None:
    if min_gap < 0:
        raise ValueError("Minimum source separation cannot be negative.")
    if not clips:
        raise ValueError("The timeline is empty.")
    highlights = 0
    for index, clip in enumerate(clips):
        number = index + 1
        if not 0 <= clip.start < clip.end <= source_frames:
            raise ValueError(f"Clip {number} lies outside the source.")
        duration = clip.frames / fps
        if clip.movie_audio and not has_audio:
            raise ValueError(f"Clip {number} needs movie audio.")
        if clip.highlight:
            highlights += 1
            if not has_audio or not 10 <= duration <= 15:
                raise ValueError(f"Highlight {number} needs movie audio and a length of 10–15 seconds.")
            parts = video_segments(clip, fps)
            if any(not 0 < part.frames / fps <= MAX_CLIP_SECONDS for part in parts):
                raise ValueError(f"Highlight {number} has an invalid segment. Each segment must be at most 5 seconds.")
        elif duration > MAX_CLIP_SECONDS:
            raise ValueError(f"Regular clip {number} exceeds 5 seconds.")
        if index:
            issue = cut_issue(clips[index - 1], clip, fps if enforce_separation else None, min_gap)
            if issue:
                raise ValueError(f"Clips {index} and {number} {CUT_MESSAGES[issue]}.")
    ordered = sorted(enumerate(clips, 1), key=lambda item: item[1].start)
    for (a, left), (b, right) in zip(ordered, ordered[1:]):
        if overlaps(left, right):
            raise ValueError(f"Clips {a} and {b} overlap or reuse the same source frames.")
    if highlights > 3:
        raise ValueError("The montage supports up to three original-audio highlight groups.")
    if final and not 960 <= sum(clip.frames for clip in clips) / fps <= 1080:
        raise ValueError("The final montage must be 16–18 minutes including highlights.")


def separation_issues(clips, fps, min_gap=DEFAULT_SOURCE_GAP):
    """Key violations by source ranges, so old edits can be repaired one cut at a time."""
    issues = {}
    for left, right in zip(clips, clips[1:]):
        gap = source_gap(left, right)
        if gap == 0 or 0 < gap / fps < min_gap:
            issues[(left.start, left.end, right.start, right.end)] = (
                "contiguous_source" if gap == 0 else "insufficient_source_gap")
    return issues


def timeline_validation_report(clips, fps, source_frames, *, has_audio=True, min_gap=DEFAULT_SOURCE_GAP,
                               fitted_duration=False):
    entries = timeline_entries(clips, fps)
    cuts = []
    for index, (left, right) in enumerate(zip(clips, clips[1:]), 1):
        issue = cut_issue(left, right, fps, min_gap)
        cuts.append({"left_edit": index, "right_edit": index + 1,
                     "source_gap_seconds": float(source_gap(left, right) / fps), "issue": issue})
    try:
        validate(clips, fps, source_frames, final=not fitted_duration, has_audio=has_audio, min_gap=min_gap)
    except ValueError as error:
        failure = str(error)
    else:
        failure = None
    ordered = sorted(enumerate(clips, 1), key=lambda item: item[1].start)
    overlapping = []
    for offset, (index, clip) in enumerate(ordered):
        for other_index, other in ordered[offset + 1:]:
            if other.start >= clip.end:
                break
            if overlaps(clip, other):
                overlapping.append([index, other_index])
    return {"passed": failure is None, "overlapping_edits": overlapping, "first_error": failure, "source_fps": str(fps),
            "minimum_source_gap_seconds": min_gap, "maximum_fragment_seconds": MAX_CLIP_SECONDS,
            "duration_fitted_to_narration": fitted_duration,
            "duration_seconds": float(sum(clip.frames for clip in clips) / fps), "cuts": cuts,
            "fragments": [{"edit": entry.edit_index + 1, "source_start_frame": entry.segment.start,
                           "source_end_frame": entry.segment.end, "timeline_start_frame": entry.position,
                           "duration_seconds": float(entry.segment.frames / fps)} for entry in entries],
            "highlight_exceptions": [{"edit": index, "source_start_frame": clip.start,
                                       "source_end_frame": clip.end, "movie_audio_enabled": clip.movie_audio}
                                      for index, clip in enumerate(clips, 1)
                                      if clip.highlight]}


def choose(candidates: list[Clip], highlights: list[Clip], fps: Fraction,
           source_frames: int, target_seconds: int, rank: ClipRanker | None = None, *, min_gap=DEFAULT_SOURCE_GAP) -> list[Clip]:
    if not 2 <= len(highlights) <= 3 or not 960 <= target_seconds <= 1080:
        raise ValueError("Select 2–3 highlights and a target of 16–18 minutes.")
    rank = rank or (lambda clip: clip.score)
    lanes: list[list[Clip]] = [[] for _ in range(12)]
    for clip in sorted(candidates, key=rank, reverse=True):
        if not any(overlaps(clip, highlight) for highlight in highlights):
            lanes[min(11, clip.start * 12 // source_frames)].append(clip)
    result: list[Clip] = []
    remaining = sorted(highlights, key=lambda clip: clip.start)
    total = 0
    target = min(int(1080 * fps), ceil(target_seconds * fps))
    maximum = int(1080 * fps)
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
    if remaining or total < ceil(960 * fps):
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
