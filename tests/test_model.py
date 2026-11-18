from dataclasses import replace
from fractions import Fraction
import unittest

from movie_review_studio.model import Clip, Signature, adjacent_conflict, choose, replacement, validate


def clip(start, length=5, image=None, highlight=False):
    value = image if image is not None else start
    signature = Signature(value, bytes([value % 256]).hex() * 256)
    return Clip(start, start + length, 1.0, (signature,), (signature,), (signature,), highlight)


class MontageTests(unittest.TestCase):
    def test_rejects_reuse_and_overlap_even_when_not_adjacent(self):
        first, middle = clip(10, image=10), clip(50, image=100)
        for repeated in (first, clip(13, image=200)):
            with self.subTest(repeated=repeated), self.assertRaisesRegex(ValueError, "overlap"):
                validate([first, middle, repeated], Fraction(1), 100)

    def test_checks_ending_frames_against_starting_frames(self):
        left = clip(10, image=10)
        right = clip(20, image=200)
        near = Signature(left.last[0].phash ^ 3, "0d" * 256)
        right = replace(right, first=(near,))
        self.assertTrue(adjacent_conflict(left, right))
        with self.assertRaisesRegex(ValueError, "look too similar"):
            validate([left, right], Fraction(1), 100)

    def test_same_interior_is_also_rejected(self):
        left, right = clip(10, image=10), clip(20, image=200)
        self.assertTrue(adjacent_conflict(left, replace(right, samples=left.samples)))

    def test_same_detected_shot_is_rejected_even_with_different_images(self):
        left = replace(clip(10, image=10), shot=4)
        right = replace(clip(20, image=200), shot=4)
        self.assertTrue(adjacent_conflict(left, right))
        self.assertFalse(adjacent_conflict(left, replace(right, shot=5)))

    def test_replacement_checks_both_neighbors_and_all_used_ranges(self):
        clips = [clip(0, image=10), clip(10, image=100), clip(20, image=200), clip(30, image=50)]
        candidates = [clip(1, image=230), clip(40, image=10), clip(50, image=200), clip(60, image=145)]
        self.assertEqual(replacement(clips, 1, candidates), candidates[-1])

    def test_regular_and_highlight_bounds(self):
        with self.assertRaisesRegex(ValueError, "exceeds 5"):
            validate([clip(0, 6)], Fraction(1), 100)
        with self.assertRaisesRegex(ValueError, "10–15"):
            validate([clip(0, 9, highlight=True)], Fraction(1), 100)
        with self.assertRaisesRegex(ValueError, "needs movie audio"):
            validate([clip(0, 10, highlight=True)], Fraction(1), 100, has_audio=False)
        with self.assertRaisesRegex(ValueError, "16–18"):
            validate([clip(0, 10, image=10, highlight=True), clip(20, 10, image=150, highlight=True)],
                     Fraction(1), 100, final=True)

    def test_generation_creates_one_valid_full_timeline(self):
        candidates = [clip(index * 6, image=index * 101 % 256) for index in range(300)]
        highlights = [clip(500, 12, 15, True), clip(1500, 12, 220, True)]
        result = choose(candidates, highlights, Fraction(1), 1800, 1020)
        validate(result, Fraction(1), 1800, final=True)
        self.assertTrue(1020 <= sum(item.frames for item in result) < 1025)
        self.assertEqual([item for item in result if item.highlight], highlights)
        for conflict in zip(result, result[1:]):
            self.assertFalse(adjacent_conflict(*conflict))

    def test_reports_insufficient_footage(self):
        highlights = [clip(0, 10, 10, True), clip(20, 10, 120, True)]
        with self.assertRaisesRegex(ValueError, "insufficient"):
            choose([], highlights, Fraction(1), 100, 960)
