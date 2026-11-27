from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from movie_review_studio.analysis import highlight_picture
from movie_review_studio.media import Media
from movie_review_studio.model import Clip, fragment_issues, replacement, timeline_validation_report, validate
from movie_review_studio.render import render
from strict_fixtures import audio_group


def part(start, end, shot=None):
    return Clip(start, end, 1, (), (), (), shot=start if shot is None else shot)


class StrictFragmentTests(TestCase):
    fps = Fraction(1)

    def test_legacy_contiguous_opening_highlight_and_ending_all_fail(self):
        for role, length in (("Opening", 10), ("Highlight", 15), ("Ending", 10)):
            with self.subTest(role=role):
                group = Clip(40, 40 + length, 0, (), (), (), highlight=True, role=role,
                             cuts=tuple(range(45, 40 + length, 5)))
                with self.assertRaisesRegex(ValueError, "touch in the source"):
                    validate([group], self.fps, 200)
                report = timeline_validation_report([group], self.fps, 200, fitted_duration=True)
                self.assertFalse(report["passed"])
                row = report["violations"][0]
                self.assertEqual((row["left"]["fragment_id"], row["right"]["fragment_id"]), ("v1", "v2"))
                self.assertEqual(row["left"]["audio_mode"], "original")
                self.assertEqual(row["source_gap_seconds"], 0)
                self.assertEqual(row["replacement_result"], "unresolved")
                self.assertEqual(report["original_audio_visual_exceptions"], 0)

    def test_group_edges_are_checked_in_both_directions_and_at_chapter_cut(self):
        group = audio_group(Clip(200, 210, 0, (), (), (), highlight=True), [(30, 35), (45, 50)])
        for edit in ([part(25, 30), group], [group, part(50, 55)],
                     [part(10, 15, 7), part(30, 35, 7)]):
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                validate(edit, self.fps, 300)
        result = [part(0, 5), group, part(55, 60)]
        validate(result, self.fps, 300, min_gap=0)
        report = timeline_validation_report(result, self.fps, 300, fitted_duration=True, min_gap=0)
        self.assertTrue(report["passed"])
        for field in ("adjacent_source_fragment_violations", "source_overlap_violations",
                      "touching_source_interval_violations", "fragments_longer_than_five_seconds",
                      "original_audio_visual_exceptions"):
            self.assertEqual(report[field], 0)

    def test_overlap_is_found_between_non_neighboring_actual_fragments(self):
        group = audio_group(Clip(70, 80, 0, (), (), (), highlight=True), [(10, 15), (30, 35)])
        edit = [group, part(50, 55), part(12, 15)]
        issues = fragment_issues(edit, self.fps, 100)
        overlap = next(row for row in issues if row["reason"] == "overlap")
        self.assertEqual((overlap["left"]["source_shot_id"], overlap["right"]["source_shot_id"]), (10, 12))
        self.assertLess(overlap["source_gap_seconds"], 0)

    def test_manual_replacement_checks_neighboring_group_picture_not_audio_range(self):
        group = audio_group(Clip(90, 100, 0, (), (), (), highlight=True), [(0, 5), (20, 25)])
        old, invalid, valid = part(45, 50), part(25, 30), part(65, 70)
        self.assertEqual(replacement([group, old], 1, [invalid, valid], fps=self.fps), valid)

    def test_final_render_rejects_internal_violation_even_for_adaptive_duration(self):
        bad = Clip(30, 45, 0, (), (), (), highlight=True, cuts=(35, 40))
        movie = SimpleNamespace(fps=self.fps, has_audio=True)
        analysis = SimpleNamespace(frames=100)
        with TemporaryDirectory() as folder, patch("movie_review_studio.render.render_clip") as decoder:
            with self.assertRaisesRegex(ValueError, "touch in the source"):
                render(movie, analysis, [bad], None, None, Path(folder), lambda _: None,
                       destination=Path(folder) / "rejected.mp4", fitted_duration=True)
            decoder.assert_not_called()
            self.assertFalse((Path(folder) / "rejected.mp4").exists())

    def test_cutaway_selector_refuses_speaker_and_same_shot_candidates(self):
        movie = Media(Path("fixture.mp4"), 96, 64, self.fps, 100, True, 0)
        anchor, same_shot, person, scenery = part(20, 25), part(12, 17, 20), part(32, 37), part(50, 55)
        analysis = SimpleNamespace(offset=0, candidates=[anchor, same_shot, person, scenery])
        # Only the scenic shot is confirmed safe by the image screening fixture.
        with patch("movie_review_studio.analysis.cutaway_without_visible_speaker",
                   side_effect=lambda _, clip, __: clip == scenery):
            self.assertEqual(highlight_picture(movie, analysis, 20, 30, None, (), 5), (anchor, scenery))
            with self.assertRaisesRegex(ValueError, "No suitable"):
                highlight_picture(movie, SimpleNamespace(offset=0, candidates=[anchor, same_shot, person]), 20, 30, None, (), 5)

