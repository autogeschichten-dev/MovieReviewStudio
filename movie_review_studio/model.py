"""Frame-indexed edit decisions and montage validation."""
from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from typing import Protocol

import numpy as np


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

    @property
    def frames(self):
        return self.end - self.start


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


def adjacent_conflict(left: Clip, right: Clip) -> bool:
    if left.shot is not None and left.shot == right.shot:
        return True
    if any(similar(a, b) for a in left.last for b in right.first):
        return True
    # Compare interior frames so alternate edits of the same shot are caught.
    return any(similar(a, b) for a in left.samples for b in right.samples)


def overlaps(left: Clip, right: Clip) -> bool:
    return left.start < right.end and right.start < left.end


def validate(clips: list[Clip], fps: Fraction, source_frames: int, *,
             final: bool = False, has_audio: bool = True) -> None:
    if not clips:
        raise ValueError("The timeline is empty.")
    highlights = 0
    for index, clip in enumerate(clips):
        number = index + 1
        if not 0 <= clip.start < clip.end <= source_frames:
            raise ValueError(f"Clip {number} lies outside the source.")
        duration = clip.frames / fps
        if clip.highlight:
            highlights += 1
            if not has_audio or not 10 <= duration <= 15:
                raise ValueError(f"Highlight {number} needs movie audio and a length of 10–15 seconds.")
        elif duration > 5:
            raise ValueError(f"Regular clip {number} exceeds 5 seconds.")
        if index and adjacent_conflict(clips[index - 1], clip):
            raise ValueError(f"Clips {index} and {number} look too similar at this cut.")
    ordered = sorted(enumerate(clips, 1), key=lambda item: item[1].start)
    for (a, left), (b, right) in zip(ordered, ordered[1:]):
        if overlaps(left, right):
            raise ValueError(f"Clips {a} and {b} overlap or reuse the same source frames.")
    if highlights > 3 or (final and highlights < 2):
        raise ValueError("The final montage requires 2–3 original-audio highlights.")
    if final and not 960 <= sum(clip.frames for clip in clips) / fps <= 1080:
        raise ValueError("The final montage must be 16–18 minutes including highlights.")


def choose(candidates: list[Clip], highlights: list[Clip], fps: Fraction,
           source_frames: int, target_seconds: int, rank: ClipRanker | None = None) -> list[Clip]:
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
        if highlight_due and (previous is None or not adjacent_conflict(previous, remaining[0])):
            selected = remaining.pop(0)
        else:
            selected = None
            for offset in range(len(lanes)):
                lane_index = (lane_number + offset) % len(lanes)
                lane = lanes[lane_index]
                for index, candidate in enumerate(lane):
                    if candidate.frames > allowance:
                        continue
                    if previous and adjacent_conflict(previous, candidate):
                        continue
                    if any(overlaps(used, candidate) for used in result):
                        continue
                    if remaining and adjacent_conflict(candidate, remaining[0]):
                        continue
                    selected = lane.pop(index)
                    lane_number = lane_index + 1
                    break
                if selected:
                    break
            if not selected:
                if remaining and (previous is None or not adjacent_conflict(previous, remaining[0])):
                    selected = remaining.pop(0)
                else:
                    break
        result.append(selected)
        total += selected.frames
    if remaining or total < ceil(960 * fps):
        raise ValueError("There is insufficient distinct usable footage to build a valid 16-minute montage.")
    validate(result, fps, source_frames, final=True)
    return result


def replacement(clips: list[Clip], index: int, candidates: list[Clip]) -> Clip:
    old = clips[index]
    used = clips[:index] + clips[index + 1:]
    left = clips[max(0, index - 1):index]
    right = clips[index + 1:index + 2]
    for clip in sorted(candidates, key=lambda item: (abs(item.frames - old.frames), -item.score)):
        if overlaps(clip, old) or any(overlaps(clip, item) for item in used):
            continue
        if not any(adjacent_conflict(neighbor, clip) for neighbor in left) and not any(
            adjacent_conflict(clip, neighbor) for neighbor in right
        ):
            return clip
    raise ValueError("No conflict-free replacement was found.")
