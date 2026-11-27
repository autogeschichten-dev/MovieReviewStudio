from fractions import Fraction
import unittest

from movie_review_studio.model import Clip, timeline_validation_report, validate


def edit(start, end, **metadata):
    return Clip(start, end, 1, (), (), (), **metadata)


class SourceSeparationTests(unittest.TestCase):
    def test_touching_fragments_are_blocked_in_both_source_orders(self):
        a, b = edit(0, 5), edit(5, 10)
        for order in ([a, b], [b, a]):
            with self.subTest(order=order), self.assertRaisesRegex(ValueError, "touch in the source"):
                validate(order, Fraction(1), 20, min_gap=0)

    def test_gap_is_exact_at_fractional_frame_rate(self):
        fps = Fraction(24000, 1001)
        left = edit(0, 100)
        with self.assertRaisesRegex(ValueError, "configured separation"):
            validate([left, edit(219, 300)], fps, 400)
        validate([left, edit(220, 300)], fps, 400)
        with self.assertRaisesRegex(ValueError, "configured separation"):
            validate([left, edit(219, 300)], fps, 400, min_gap=4)

    def test_actual_source_shot_identity_survives_a_temporal_gap(self):
        with self.assertRaisesRegex(ValueError, "same continuous source shot"):
            validate([edit(0, 4, shot=3), edit(30, 34, shot=3)], Fraction(1), 50)

    def test_highlight_has_no_internal_exception(self):
        highlight = edit(0, 12, highlight=True, cuts=(4, 8))
        with self.assertRaisesRegex(ValueError, "touch in the source"):
            validate([highlight, edit(17, 22)], Fraction(1), 30)
        report = timeline_validation_report([highlight, edit(17, 22)], Fraction(1), 30)
        self.assertEqual(report["original_audio_visual_exceptions"], 0)
        self.assertEqual(report["touching_source_interval_violations"], 2)
        self.assertEqual(report["violations"][0]["source_gap_seconds"], 0)
        self.assertEqual([row["duration_seconds"] for row in report["fragments"]], [4, 4, 4, 5])

    def test_report_identifies_gap_and_final_duration_failures(self):
        edits = [edit(0, 4), edit(6, 10)]
        result = timeline_validation_report(edits, Fraction(1), 30)
        self.assertFalse(result["passed"])
        self.assertEqual(result["violations"][0]["reason"], "insufficient_source_gap")
        result = timeline_validation_report(edits, Fraction(1), 30, min_gap=2)
        self.assertEqual(result["violations"][0]["reason"], "insufficient_source_gap")
        self.assertEqual(result["minimum_source_gap_seconds"], 5)
